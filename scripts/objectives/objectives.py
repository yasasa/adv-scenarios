import os
from tkinter import W
from cubeadv.sim.sensors.ngp_lidar import InstantNGPLidar
from re import X
import torch
import numpy as np
import imageio
import time
from dataclasses import dataclass
from typing import Union, List, Dict, AnyStr

from cubeadv.fields import NGPField
from cubeadv.opt import discrete_adjoint
from cubeadv.sim.dynamics import Dynamics
from cubeadv.sim.utils import PathMapCost, STARTS

#from cubeadv.policies.depth_policy import RGBNet, Policy
from cubeadv.policies.color_policy import RGBNet, Policy
from cubeadv.policies.cnn import PolicyNoStart

from cubeadv.utils import normalize, make_functional, set_weights, get_nerf_max, get_nerf_min

from scripts.utils import get_bov_image
from scripts.policy_train_utils import build_nerf_lidar
        

#import functorch


class Objective:
    """ Objective to be maximized """
    def __init__(self, cfg):
        self.start = STARTS[cfg.start].expand(cfg.opt_batch, -1).to(cfg.device)
        self.goal = STARTS[cfg.goal].expand(cfg.opt_batch, -1).to(cfg.device)
        self.batch = cfg.opt_batch

        self.x0 = torch.cat([self.start, torch.zeros(self.batch, 1).to(cfg.device)], dim=-1)

        self.goal_one_hot = torch.nn.functional.one_hot(torch.tensor([cfg.goal]*cfg.opt_batch), len(STARTS)).view(cfg.opt_batch, -1).to(cfg.device)

        policy_channels_in = 4 if (not cfg.no_depth) else 3

        self.cfg = cfg
        self.device = torch.device(cfg.device)

        if cfg.ngp_field:
            self.policy = PolicyNoStart()
            self.policy.load_state_dict(torch.load(cfg.policy_model_path))
            self.policy = self.policy.to(self.device)
        else:
            net_policy = RGBNet(num_points=cfg.lidar_num_points, channels_in=policy_channels_in)
            net_policy.load_state_dict(torch.load(cfg.policy_model_path))
            net_policy = net_policy.to(self.device)
            self.policy = Policy(net_policy, rgb=True, num_points=cfg.lidar_num_points, height=cfg.lidar_num_channels, channels_in=policy_channels_in)

        if not cfg.single_frame:
            self.init_multiframevars()

    def init(self):
        pass

    def set_device(self, device):
        raise ValueError("This objective function doesn't support changing devices")

    def init_multiframevars(self):
        # Trajectory params
        self.T = self.cfg.dt * torch.arange(self.cfg.num_steps_traj).to(self.device)

        # Dynamics
        self.dynamics = Dynamics(self.cfg.dt) # Note: 0.001 here is obsolete & unused

        # Cost function for trajectory opt
        # Note: pm.cost does not have a negative sign in front since this is
        # the objective being maximized.

        self.pm = PathMapCost.get_carla_town(self.start.cpu(), self.goal.cpu()).to(self.device)
        if self.cfg.car_start is None:
            x0_ = self.start
            if self.cfg.multistart > 0.1:
                offsets = self.cfg.multistart*(torch.rand(self.batch, 1) * 2 - 1).to(self.device)
                x0_ = self.pm.get_offset(self.start, offsets)
            self.x0 = torch.cat([x0_, self.pm.get_initial_lane_alignment().view(self.batch, -1)], dim=-1)
        else:
            self.x0  = torch.tensor(self.cfg.car_start).view(self.batch, 3).cuda()

        self.cost_fn = lambda x: self.pm.cost(x[:, :2], None)

    def sensor_step(self, x, p, i=0):
        return self.sensor(x, p)

    def policy_step(self, x, p, i=0):
        x_ = x.clone()
        if not self.cfg.dont_detach_yaw:
            x_[:, 2] = x[:, 2].detach()

        if self.cfg.detach:
            o = self.sensor_step(x_.detach(), p, i)
        else:
            o = self.sensor_step(x_, p, i)

        #u = self.policy(o)
        u = self.policy(o, self.goal_one_hot)

        return u

    def F(self, i, t2, t1, x2, x1, p):
        x1 = x1.view(self.batch, -1)
        u = self.policy_step(x1, p, i).view(self.batch)
        xdot = self.dynamics.f(x1.view(self.batch, -1), u)
        x2 = x1 + xdot*(t2 - t1)
        return x2

    def objective_multiframe(self, p, ret_traj=False, reduce=True):
        cost, xs = discrete_adjoint(self.F, self.cost_fn, self.x0, self.T, p, ret_traj=True)
        if reduce:
            cost = cost.mean(0)
        if ret_traj:
            return cost, xs.mean(dim=0)
        else:
            return cost

    def objective(self, p, ret_traj=False):
        raise NotImplementedError()

    def get_random_params(self):
        raise NotImplementedError()

    def get_debug_vars(self, itr, p):
        return None

    def get_constraints(self):
        return torch.tensor(0), torch.tensor(1.)

    def __call__(self, p, ret_traj=False):
        return self.objective(p, ret_traj)

    def render(self, p):
        return None, None

    def render(self, p):
        pass

    def render_pre(self):
        pass

# ============================== Cube Objective ===============================

class CubeObjective(Objective):
    """ Outputs policy """
    def __init__(self, cfg):
        """
        Args:
            cfg: Experiment configuration.
        """
        super().__init__(cfg)
        self.PARAMS_PER_CUBE = 12
        self.box_num = cfg.num_cubes
        if cfg.init_params is not None:
            assert cfg.num_cubes == torch.load(cfg.init_params).shape[0] // self.PARAMS_PER_CUBE
        self.sensor = build_nerf_lidar(cfg, self.box_num)

    def get_constraints(self):
        num_cubes = self.box_num

        #min_bounds = torch.cat((- 1.0 * torch.ones(num_cubes, 6), 0.0 * torch.ones(num_cubes, 3), -30.0 * torch.ones(num_cubes, 3)), dim=1).flatten()
        #max_bounds = torch.cat((1.0 * torch.ones(num_cubes, 6), 1.0 * torch.ones(num_cubes, 3), 30.0 * torch.ones(num_cubes, 3)), dim=1).flatten()

        # Define in meters (real coordinates)
        min_xyz = [80.0, 110.0, 0.0]
        max_xyz = [110.0, 140.0, 9.0]
        min_c, max_c = 0.0, 1.0
        min_s, max_s = 0.5, 3.0
        min_a, max_a = -30.0, 30.0
        bounds = [(min_xyz[0] + max_s/2, max_xyz[0] - max_s/2),
                  (min_xyz[1] + max_s/2, max_xyz[1] - max_s/2),
                  (min_xyz[2] + max_s/2, max_xyz[2] - max_s/2),
                  * ((min_c, max_c),) * 3,
                  * ((min_s, max_s),) * 3,
                  * ((min_a, max_a),) * 3]

        min_bounds, max_bounds = zip(*bounds)
        min_bounds = np.tile(np.array(min_bounds, dtype=np.float32), num_cubes)
        max_bounds = np.tile(np.array(max_bounds, dtype=np.float32), num_cubes)

        # Convert to nerf coordinates
        min_bounds = normalize(torch.from_numpy(min_bounds), get_nerf_max(), get_nerf_min())
        max_bounds = normalize(torch.from_numpy(max_bounds), get_nerf_max(), get_nerf_min())

        return min_bounds, max_bounds

    def get_random_params(self):
        min_bounds, max_bounds = self.get_constraints()
        rand_params = min_bounds + torch.rand(self.PARAMS_PER_CUBE * self.box_num) * (max_bounds - min_bounds)
        return rand_params


class SingleFrameCube(CubeObjective):
    def objective(self, p, ret_traj=False):
        o = self.sensor(self.x0, p)
        u = self.policy(o)
        return u

class MultiFrameCube(CubeObjective):
    """ Outputs policy """
    def __init__(self, cfg):
        """
        Args:
            cfg: Experiment configuration.
        """
        super().__init__(cfg)

    def objective(self, p, ret_traj=False):
        return self.objective_multiframe(p, ret_traj)

# ============================== Voxel Objective ===============================

class SingleFrameVoxel(Objective):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.sensor = build_nerf_lidar(cfg)
        self.params = self.__init_params()

    def objective(self, p, ret_traj=False):
        o = self.sensor(self.x0, None)

        #im_show = o.transpose(0, 1)
        #im_show = im_show.cpu().detach().numpy()
        #im_show = (im_show * 255).astype(np.uint8)
        #imageio.imwrite('tmp_perturbed2.png',im_show)

        u = self.policy(o)
        return u

    def __init_params(self):
        grid = self.sensor._world
        current_params = []
        for name, param in grid.named_parameters():
            if param.requires_grad and name == 'sh_data': #(name == 'density_data' or name == 'sh_data'):
                print(name) #, param.data)
                current_params.append({'params': param})

        # Plenoxel parameters:
        #grid.sh_data
        #grid.density_data
        #grid.basis_data
        #grid.background_data

        return current_params

class MultiFrameVoxel(Objective):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.sensor = build_nerf_lidar(cfg)
        self.meta, self.original_params = self.__init_params()
        self.params = torch.zeros(self.original_params.shape, device=self.original_params.device, requires_grad=True)

    def sensor_step(self, x, p):
        set_weights(self.sensor._world, self.meta, self.original_params + p)
        o = self.sensor(x, None)

        # im_show = o.transpose(0, 1)
        # im_show = im_show.cpu().detach().numpy()
        # im_show = (im_show * 255).astype(np.uint8)
        # imageio.imwrite('tmp_perturbed2.png',im_show)

        return o

    def objective(self, p, ret_traj=False):
        return self.objective_multiframe(p, ret_traj)

    def __init_params(self):
        grid = self.sensor._world
        param_filter = lambda name: name == 'sh_data' or name == 'density_data'
        #param_filter = None
        params, meta = make_functional(grid, param_filter=param_filter, verbose=True)
        current_params = torch.cat([_p.data.flatten() for _p in params])
        print('Parameter shape:', current_params.shape)
        return meta, current_params

# ======================== NGP Colour Attack Objective =========================

class SingleFrameNGPColourAttack(Objective):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.sensor = build_nerf_lidar(cfg)
        self.params = self.__init_params()

    def objective(self, p, ret_traj=False):
        o = self.sensor(self.x0, None)

        debug = False
        if debug:
            self.sensor.save_img(o.transpose(0,1)[:,:,0:3], "tmp_rgb_out0.png")
            print('saved sensor out to tmp_rgb_out0.png')

        u = self.policy(o)
        return u

    def __init_params(self):
        current_params = []
        for pipeline_obj in self.sensor.pipeline_objs:
            nef = pipeline_obj.nef
            for name, param in nef.named_parameters():
                if param.requires_grad and 'decoder_color' in name:
                    print("Adding parameter to attack:", name)
                    current_params.append({'params': param})
        return current_params

    def render(self, p):
        output = self.sensor(self.x0, None)
        return output[:, :, :3].transpose(0,1)


class MultiFrameParamAttack(Objective):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.field, self.sensor = build_nerf_lidar(self.cfg)
        if len(cfg.obj_fields) > 0:
            for field_cfg in cfg.obj_fields:
                field = NGPField(field_cfg)
                self.field.add_obj_field(field)
        else:
            for i in range(cfg.num_obj):
                field = NGPField(cfg.obj_cfg_path)
                self.field.add_obj_field(field)

        self.param_info, self.param_vector, self.param_split = self.init_texture_params()
        self.texture_params = torch.zeros_like(self.param_vector).requires_grad_(True)

        if cfg.transform_params is not None:
            self.transform_params = torch.tensor(cfg.transform_params).cuda()
        else:
            self.transform_params = ComposeObjective.get_random_params(self)
            filename = os.path.join(cfg.output_dir, "{}-log.pt".format("transform_params"))
            torch.save(self.transform_params, filename)
        self.field.set_transform_params(self.transform_params)
        self.set_texture_params(self.texture_params)

    def init_texture_params(self):
        param_filter = lambda name: self.cfg.param_search_keyword in name
        obj_params = []

        for obj in self.field.obj_fields:
            params = make_functional(obj.pipeline.nef, param_filter=param_filter, verbose=False)
            obj_params.append(params)

        param_v = torch.cat([_p.param_vector for _p in obj_params])
        param_sizes = [_p.param_vector.numel() for _p in obj_params]
        print(f"TOTAL PARAMETER COUNT: {sum(param_sizes)}")

        return obj_params, param_v, param_sizes

    def set_texture_params(self, p):
        assert(p.dim() == 1 or p.shape[0] == 1)
        p = p.squeeze()
        assert(p.shape[0] == self.param_vector.shape[0])

        updated_params = self.param_vector + p
        updated_params = updated_params.split(self.param_split)

        for i, field in enumerate(self.field.obj_fields):
            set_weights(field.pipeline.nef, self.param_info[i], updated_params[i])

    def get_random_texture_params(self):
        return torch.zeros_like(self.texture_params).requires_grad_(True)

    def objective(self, p, ret_traj=False):
        x, t = self.objective_multiframe(p, ret_traj=True)
        self.cached_traj = t.cpu().detach()
        return x

    def plot(self, p):
        if self.cached_traj is None:
            self.objective(p)
        self.set_texture_params(p)
        fig, ax = get_bov_image(self.cached_traj, self.pm, fields=self.field.obj_fields, obj_transforms=self.transform_params.view(-1, 4))
        return fig, ax

    def render(self, p):
        im = self.sensor_step(self.x0, p)[0].permute(1, 2, 0)
        return im

class MultiFrameNGPColourAttack(MultiFrameParamAttack):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.params = self.texture_params

    def get_random_params(self):
        return self.get_random_texture_params()

    def sensor_step(self, x, p, i=0):
        self.set_texture_params(p)
        o, _ = self.sensor(self.field, x)
        o = o.permute(0, 3, 1, 2)
        return o

class RandomColourAttack(MultiFrameNGPColourAttack):
    def __init__(self, cfg):
        super().__init__(cfg)

    def get_constraints(self):
        return torch.tensor(-5.), torch.tensor(5.)

class BOColourAttack(RandomColourAttack): # Just changing the name for config sake
    def __init__(self, cfg):
        super().__init__(cfg)

class PerturbationOnReference(MultiFrameParamAttack):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.params = self.texture_params
        self.ref_t = None
        _, self.ref_t = self.objective_multiframe(self.params, ret_traj=True)

    def get_random_params(self):
        return self.get_random_texture_params()

    def sensor_step(self, x, p, i=0):
        if self.ref_t is None:
            x_s = x
        else:
            x_s = self.ref_t[i]
        self.set_texture_params(p)
        o, _ = self.sensor(self.field, x_s)
        o = o.permute(0, 3, 1, 2)
        return o

class MultiFrameNGPTransformAttack(MultiFrameParamAttack):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.base_transform_params = self.transform_params.view(-1, 4)
        self.params = self.get_random_params()


    def get_random_params(self):
        params = torch.zeros_like(self.base_transform_params).requires_grad_(True)
        return params

    def sensor_step(self, x, p, i=0):
        p = p.squeeze()
        assert(p.shape == self.base_transform_params.shape)

        p = p.clone()

        p[:, :2] += self.base_transform_params[:, :2]
        p[:, 2:] = self.base_transform_params[:, 2:]

        o, _ = self.sensor(self.field, x, obj_transform=p.view(-1))
        o = o.permute(0, 3, 1, 2)
        return o

# ======================== Object composition Objective ========================

class ComposeObjective(Objective):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.ngp_field:
            self.field, self.sensor = build_nerf_lidar(self.cfg)
        else:
            self.sensor = build_nerf_lidar(cfg)

    def get_random_params(self):
        rand_params = torch.rand(3*self.cfg.num_obj).to(self.device) - 0.5 # translation
        #for i in range(0, rand_params.shape[0], 3):
        #    rand_params[i+1] *= 10
        #    rand_params[i+2] *= 0.5
        return rand_params[None, ...]
        #return torch.randn(1) + 0.5 # scale


class SingleFrameCompose(ComposeObjective):
    def __init__(self, cfg):
        super().__init__(cfg)

    def objective(self, p, ret_traj=False):
        o = self.sensor(self.x0, p)
        # self.sensor.save_img(o.transpose(0,1)[:,:,0:3], "tmp_rgb_out0.png")
        # print('saved sensor out to tmp_rgb_out0.png')
        u = self.policy(o)
        return u

    def render(self, p):
        output = self.sensor(self.x0, p)
        return output[:, :, :3].transpose(0,1)

class MultiFrameCompose(ComposeObjective):
    def __init__(self, cfg):
        super().__init__(cfg)

    def sensor_step(self, x, p, i=0):
        if self.cfg.ngp_field:
            self.field.set_transform_params(p)
            o, _ = self.sensor(self.field, x)
        else:
            o = self.sensor(x, self.transform_params)

        debug = False
        if debug:
            if self.cfg.ngp_field:
                self.sensor.save_img(o[0], "tmp_rgb_out0.png")
            else:
                self.sensor.save_img(o.transpose(0,1)[:,:,0:3], "tmp_rgb_out0.png")
            print('saved sensor out to tmp_rgb_out0.png')
            exit()

        if self.cfg.ngp_field:
            o = o.permute(0, 3, 1, 2)

        return o

    def objective(self, p, ret_traj=False):
        x, t = self.objective_multiframe(p, ret_traj=True)
        self.cached_traj = torch.stack(t).detach().cpu()
        return x

    def plot(self, p):
        if self.cached_traj is None:
            self.objective(p)

        fig, ax = get_bov_image(self.cached_traj, self.pm)
        return fig, ax

    def render(self, p):
        im = self.sensor_step(self.x0, p[0])[0].permute(1, 2, 0)
        return im


# ======================== Pixel Perturbation Objective ========================

class PerturbObjective(Objective):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.sensor = build_nerf_lidar(cfg)

    def get_random_params(self):
        channels = 3 if self.cfg.no_depth else 4
        return torch.randn(self.cfg.lidar_num_points//self.cfg.lidar_num_channels,
                           self.cfg.lidar_num_channels, channels).to(self.device)

class SingleFramePerturb(PerturbObjective):
    def __init__(self, cfg):
        super().__init__(cfg)

    def objective(self, p, ret_traj=False):
        o = self.sensor(self.x0, None) + p
        u = self.policy(o)
        return u

    def render(self, p):
        output = self.sensor(self.x0, None) + p
        return output[:, :, :3].transpose(0, 1)#.permute(2, 1, 0)

class MultiFramePerturb(PerturbObjective):
    def __init__(self, cfg):
        super().__init__(cfg)

    def sensor_step(self, x, p, i=0):
        o = self.sensor(x, None)

        debug = False
        if debug:
            self.sensor.save_img(o.transpose(0,1)[:,:,0:3], "tmp_rgb_out0.png")
            print('saved sensor out to tmp_rgb_out0.png')

        return o + p

    def objective(self, p, ret_traj=False):
        return self.objective_multiframe(p, ret_traj)

class SingleFrameRefPerturb(Objective):
    """ Outputs policy """
    def __init__(self, cfg):
        """
        Args:
            cfg: Experiment configuration.
        """
        super().__init__(cfg)
        self.ref_image = self.__get_ref_image(cfg, self.x0).detach()

    def __get_ref_image(self, cfg, x):
        sensor = build_nerf_lidar(cfg)
        return sensor(x, torch.ones(12 * 5).cuda())

    def get_random_params(self):
        return torch.randn(100, 32, 4)

    def objective(self, p, ret_traj=False):
        o = self.ref_image + p
        u = self.policy(o)
        return u

# =============================================================================

class SingleFrameNerfTexture(Objective):
    """ Outputs policy """
    def __init__(self, cfg):
        """
        Args:
            cfg: Experiment configuration.
        """
        super().__init__(cfg)

        tmp_c0 = torch.load(cfg.cube_param_path)
        tmp_c0 = tmp_c0.view(-1, 12)
        tmp_c0[:, 3:6] = 1. # no color bias in these
        self.c0 = tmp_c0.flatten().cuda()
        self.box_num = tmp_c0.shape[0]

        self.sensor = build_nerf_lidar(cfg, self.box_num, independ_boxnet=True)

        (self.network,
         self.params,
         self.meta,
         self.current_params) = self.__init_nerf_params(self.sensor)

    def get_random_params(self):
        return torch.randn_like(self.current_params)

    def __init_nerf_params(self, sensor):
        network = sensor._world.models['net_0'].module.nerf_net
        model_param_filter = lambda name: name.startswith("box_net") and "rgb_layers" in name
        params, meta = make_functional(network, model_param_filter)
        current_params = torch.cat([_p.data.flatten() for _p in params])

        return network, params, meta, current_params

    def __render_camera(self, p):
        cam_builder = lambda _, x, y: make_sim(self.cfg.nerf_config, 340, 680,
                                               chunk_size=16384, camera=True,
                                               box_num=self.box_num)
        cam_sensor = FreeCameraSensor(cam_builder)
        network, params, meta, current_params = self.__init_nerf_params(cam_sensor)
        set_weights(network, meta, p)
        angles = [-np.pi, -np.pi/2, 0, np.pi/2]
        with torch.no_grad():
            rgbs = []
            for angle in angles:
                x0 = self.x0.clone()
                x0[2] = angle
                rgb = cam_sensor.read(x0, self.c0)
                rgb = rgb.detach().cpu().numpy()
                rgbs.append(rgb)
        return rgbs

    def objective(self, p, ret_traj=False):
        set_weights(self.network, self.meta, p)
        o = self.sensor(self.x0, self.c0)
        u = self.policy(o)
        return u

    def get_debug_vars(self, itr, p):
        return self.__render_camera(p)

class NGPGridAttack(Objective):
    def __init__(self, cfg):
        super().__init__(cfg)
        assert(cfg.ngp)

    def init(self):
        self.sensor = build_nerf_lidar(self.cfg)
        self.grid = self.sensor._world.nef.grid
        self.grid_sizes = [p.numel() for p in self.grid.codebook]
        self.base_params, self.meta = make_functional(self.grid)

        self.params = self.get_random_params()

    def set_device(self, device):
        # TODO.YA: Overkill
        self.sensor._world = self.sensor._world.to(device)
        self.params = self.params.to(device)
        self.sensor._world.nef = self.sensor._world.nef.to(device)
        self.sensor._world.tracer = self.sensor._world.tracer.to(device)
        self.grid = self.grid.to(device)
       # self.grid.blas.set_device(device) # custom function
        self.x0 = self.x0.to(device)
        self.policy._net = self.policy._net.to(device)
        self.device = device

    def _get_sensor_output(self, p):
        _p = torch.cat([p_.flatten() for p_ in p])
        set_weights(self.grid, self.meta, _p)
        o = self.sensor(self.x0, None)
        return o

    def objective(self, p, ret_traj=False):
       # u = self.policy(self._get_sensor_output(p))
        o = self._get_sensor_output(p).reshape(-1, 4)
        return o[:, 2].sum()

    def render(self, p):
        output = self._get_sensor_output(p)
        return output.transpose(0, 1)[:, :, :3]

    def get_random_params(self):
        params = [p.detach().clone() + torch.randn_like(p) for p in self.base_params]
        return params

class NGPFunctionalAttack(NGPGridAttack):
    def init(self):
        self.sensor = build_nerf_lidar(self.cfg)
        self.f, self.base_params = functorch.make_functional(self.sensor)
        self.params = self.base_params

    def set_device(self, device):
        # TODO.YA: Overkill
        self.sensor._world = self.sensor._world.to(device)
        self.params = [p.to(device) for p in self.params]
        self.sensor._world.nef = self.sensor._world.nef.to(device)
        self.sensor._world.tracer = self.sensor._world.tracer.to(device)
        self.grid = self.grid.to(device)
       # self.grid.blas.set_device(device) # custom function
        self.x0 = self.x0.to(device)
        self.policy._net = self.policy._net.to(device)
        self.device = device


    def _get_sensor_output(self, p):
        o = self.f(p, self.x0, None)
        return o

class NGPGridPolicyAttack(NGPGridAttack):
    def objective(self, p, ret_traj=False):
        u = self.policy(self._get_sensor_output(p))
        return -u.abs()

class NGPGridMaximizeRedDifferential(NGPGridAttack):
    def objective(self, p, ret_traj=False):
        o = self._get_sensor_output(p)
        return o[:, :, 0].sum() - o[:, :, 1:3].sum()

class NGPFunctionalPolicyAttack(NGPFunctionalAttack):
    def objective(self, p, ret_traj=False):
        u = self.policy(self._get_sensor_output(p))
        return -u.abs()

class NGPFuncMaximizeRedDifferential(NGPGridAttack):
    def objective(self, p, ret_traj=False):
        o = self._get_sensor_output(p)
        return o[:, :, 0].sum() - o[:, :, 2].sum()

class PoseOptimization(Objective):
    def __init__(self, cfg):
        """
        Args:
            cfg: Experiment configuration.
        """
        super().__init__(cfg)
        self.sensor = build_nerf_lidar(cfg)
        self.p0 = self.x0[:2]
        self.ref = self._get_sensor_output(self.p0).detach()
        self.params = self.get_random_params()

    def _get_sensor_output(self, p):
        pn = torch.zeros_like(self.x0)
        pn[:2] = p
        #print(pn)
        return self.sensor(pn, None)

    def objective(self, p, ret_traj=False):
        e = (self.ref - self._get_sensor_output(p)).flatten()
        return -e.dot(e)

    def render(self, p):
        output = self._get_sensor_output(p)
        return output.transpose(0, 1)[:, :, :3]

    def render_pre(self):
        return self.render(self.x0[:2])

    def get_random_params(self):
        return self.p0 + torch.randn_like(self.p0)

# =============================================================================

class NGPFieldAttack(Objective):
    def __init__(self, cfg):
        super().__init__(cfg)
        assert(cfg.ngp_field)

    def init(self):
        self.field, self.sensor = build_nerf_lidar(self.cfg)
        self.params = self.field.parameters()

    def set_device(self, device):
        # TODO.YA: Overkill
        self.sensor._world = self.sensor._world.to(device)
        self.params = self.params.to(device)
        self.sensor._world.nef = self.sensor._world.nef.to(device)
        self.sensor._world.tracer = self.sensor._world.tracer.to(device)
        self.grid = self.grid.to(device)
       # self.grid.blas.set_device(device) # custom function
        self.x0 = self.x0.to(device)
        self.policy._net = self.policy._net.to(device)
        self.device = device

    def sensor_step(self, x, p, i=0):
        o, _ = self.sensor(self.field, x.view(self.batch, -1).detach())

        debug = False
        if debug:
            self.sensor.save_img(o[0], "tmp_rgb_out0.png")
            print('saved sensor out to tmp_rgb_out0.png')
            exit()

        o = o.permute(0, 3, 1, 2)
        o = o + p
        return o

    def plot(self, p):
        if self.cached_traj is None:
            self.objective(p)

        fig, ax = get_bov_image(self.cached_traj, self.pm)
        return fig, ax

    def render(self, p):
        im = self.sensor_step(self.x0, p)[0].permute(1, 2, 0)
        return im

    def get_random_params(self):
        return 0.0001*torch.randn(self.batch, *self.policy.requested_feature_shape).to(self.device)

    def objective(self, p, ret_traj=False):
        x, t = super().objective_multiframe(p, ret_traj=True)
        self.cached_traj = t.cpu().detach()
        return x

class SteeringMaximizationObjective(MultiFrameNGPColourAttack):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def simulate(self, x, p):
        xs = []
        cost = torch.zeros(x.shape[0]).type_as(p)
        for i in range(self.cfg.num_steps_traj):
            def step(x, p):
                x = x.view(self.batch, -1)
                u = self.policy_step(x, p, i).view(self.batch)
                xdot = self.dynamics.f(x.view(self.batch, -1), u)
                x = x + xdot*self.cfg.dt
                return x, -torch.abs(-1. - u)
            x, c = torch.utils.checkpoint.checkpoint(step, x, p)
            xs.append(x.clone())
            cost += c
        return cost, torch.cat(xs, dim=0)[None, ...]
        
    def get_random_params(self):
        return super().get_random_texture_params()

    def objective_multiframe(self, p, ret_traj=False, reduce=True):
        print(p.requires_grad)
        cost, xs = self.simulate(self.x0, p)
        print(xs.shape)
        
        if reduce:
            cost = cost.mean(0)
        if ret_traj:
            return cost, xs.mean(dim=0)
        else:
            return cost
