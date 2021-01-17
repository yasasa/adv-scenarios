import torch
from scripts.objectives.base_objectives import Objective
from utils import get_sensor_from_cfg


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
        self.sensor = get_sensor_from_cfg(cfg)

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
        self.sensor = get_sensor_from_cfg(cfg)
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