import torch

from .base_objectives import Objective
from .utils import get_bov_image, get_sensor_from_cfg

from cubeadv.fields import NGPField, NGPComposeField
from cubeadv.utils.torch_utils import make_functional, set_weights


class SingleFrameNGPColourAttack(Objective):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.sensor = get_sensor_from_cfg(cfg)
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

class NGPParamAttack(Objective):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.field, self.sensor = get_sensor_from_cfg(self.cfg)
        if len(cfg.obj_fields) > 0:
            for field_cfg in cfg.obj_fields:
                field = NGPField(field_cfg, scene_midpoint=torch.zeros(3), scene_scale=torch.ones(3))
                self.field.add_obj_field(field)
        else:
            for i in range(cfg.num_obj):
                field = NGPField(cfg.obj_cfg_path, scene_midpoint=torch.zeros(3), scene_scale=torch.ones(3))
                self.field.add_obj_field(field)

        self.param_info, self.param_vector, self.param_split = self.init_texture_params()
        self.texture_params = torch.zeros_like(self.param_vector).requires_grad_(True)

        self.transform_params = torch.tensor(cfg.transform_params).cuda()
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
        
    def get_color_constraints(self):
        pv = self.param_vector.clone()
        return  -5 * torch.ones_like(pv), 5*torch.ones_like(pv)
    
    def set_params(self, p):
        self.set_texture_params(p)

    def set_texture_params(self, p):
        assert(p.dim() == 1 or p.shape[0] == 1)
        p = p.squeeze()
        assert(p.shape[0] == self.param_vector.shape[0])

        updated_params = self.param_vector + p
        updated_params = updated_params.split(self.param_split)

        for i, field in enumerate(self.field.obj_fields):
            set_weights(field.pipeline.nef, self.param_info[i], updated_params[i])

    def get_random_texture_params(self):
        return torch.randn_like(self.texture_params).requires_grad_(True) * 1e-4

    def objective(self, p, ret_traj=False):
        x, t = self.objective_multiframe(p, ret_traj=True)
        self.cached_traj = t.cpu().detach()
        return x

    def plot(self, p):
        if self.cached_traj is None:
            self.objective(p)
        self.set_params(p)
        fig, ax = get_bov_image(self.cached_traj, self.pm, fields=self.field.obj_fields, obj_transforms=self.transform_params.view(-1, 4))
        return fig, ax

    def render(self, p):
        im = self.sensor_step(self.x0, p)[0].permute(1, 2, 0)
        return im

class MultiFrameNGPColourAttack(NGPParamAttack):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.params = self.texture_params

    def get_random_params(self):
        return self.get_random_texture_params()

    def sensor_step(self, x, p, i=0):
        self.set_params(p)
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

class MultiFrameNGPTransformAttack(NGPParamAttack):
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

class FrameLevelAttack(MultiFrameNGPColourAttack):
    """Fixed frame wise attack baseline"""
    def __init__(self, cfg):
        super().__init__(cfg)

        # Hard coding the reference control:
        self.u_ref = 1.

    def sensor_step(self, x, p):
        o, _ = self.sensor(self.field, x)
        return o.permute(0, 3, 1, 2)

    def simulate(self, p, ret_traj):
        self.set_params(p)
        x = self.x0
        us = []
        xs = [x.clone()]

        def step(x, tn, t, p):
            o = self.sensor_step(x.detach(), p)
            u = self.policy(o, self.goal_one_hot).squeeze()
            xn = x + (tn - t) * self.dynamics.f(x.view(self.batch, -1),
                                                u[None])
            return xn, u

        for tn, t in zip(self.T[1:], self.T[:-1]):
            xn, u = torch.utils.checkpoint.checkpoint(step, x, tn, t, p)
            xs.append(xn.clone())
            us.append(u)
            x = xn

        u = torch.stack(us).squeeze()
        cost = -torch.sum((self.u_ref - u)**2)

        if ret_traj:
            return cost, torch.stack(xs).squeeze()
        return cost

    def objective(self, p, ret_traj=False):
        x, t =  self.simulate(p, True)
        self.cached_traj = t.cpu().detach()
        return x

class NGPTransformAttack(NGPParamAttack):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        if cfg.transform_min is not None:
            self.min = torch.tensor(cfg.transform_min).type_as(self.texture_params)
        
        if cfg.transform_max is not None:
            self.max = torch.tensor(cfg.transform_max).type_as(self.texture_params)
  #      self.o_ref = self.sensor_step(self.x0, self.transform_params + torch.randn_like(self.transform_params)).detach()
        
    def get_random_params(self):
        return self.get_random_transform_params()

    def get_random_transform_params(self):
        params = self.transform_params.clone()
        params = torch.rand_like(params) * (self.max - self.min) + self.min
   #     params += 1e-4 * torch.randn_like(params)
        params = params.clone().requires_grad_(True)
        return params

    @torch.no_grad()
    def plot_loss(self, p0, res, dim):
        min_v = self.min[dim]
        max_v = self.max[dim]

        values = torch.linspace(min_v, max_v, res)
        obj_v = []
        for v in values:
            p_ = p0.clone()
            p_[dim] = v
            obj_v.append(self.objective(p_))
            
        return values, torch.stack(obj_v)


    def set_params(self, p):
        print(p.shape)
        self.field.set_transform_params(p)

    def sensor_step(self, x, p, i=0):
        o, _ = self.sensor(self.field, x, obj_transform=p)
        return o.permute(0, 3, 1, 2)

    def project_to_constraints(self, x):
        return x.clamp(min=self.min, max=self.max)

    def constraint(self, x):
        return torch.tensor(0.).type_as(x) # No constraints

    def get_constraints(self):
        return self.min, self.max
        
    def get_transform_constraints(self):
        return self.transform_params - 1, self.transform_params + 1
        
class TransformAndColorAttack(NGPTransformAttack):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.tr_param_len = self.transform_params.shape[-1]
        self.params_base = torch.cat([self.transform_params, self.texture_params]).detach()

    def set_params(self, p):
        tr_param_len = self.transform_params.shape[-1]
        transform, texture = p[:tr_param_len], p[tr_param_len:]
        self.field.set_transform_params(transform)
        self.set_texture_params(texture)

    def sensor_step(self, x, p, i=0):
        tr_param_len = self.transform_params.shape[-1]
        self.set_texture_params(p[tr_param_len:])
        o, _ = self.sensor(self.field, x, obj_transform = p[:tr_param_len])
        return o.permute(0, 3, 1, 2)

    def project_to_constraints(self, x):
        transform = x
        transform = super().project_to_constraints(transform)
        return transform

    def constraint(self, x):
        return torch.tensor(0.).type_as(x) # No constraints

class MixedTransformAttack(TransformAndColorAttack):
    def __init__(self, cfg):
        super().__init__(cfg)

class RandomTransformAttack(TransformAndColorAttack):
    def __init__(self, cfg):
        super().__init__(cfg)