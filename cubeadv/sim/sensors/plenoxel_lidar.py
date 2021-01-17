import sys
import argparse
import time
import copy
import imageio

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from .raw_sensor import RawSensor
from ..simple_sim import UNet
from .sensor_utils import get_lidar_rays, yaw_to_mat
#try:
#    import svox2
#    import svox2.utils
#except ImportError:
#    print("Plenoxels not installed")
sys.path.append("..")
from plenoxels.svox2.opt.util import config_util
from plenoxels.svox2.opt.util.dataset import datasets
from cubeadv.utils import normalize_pose_xy
import plenoxels.svox2.svox2 as svox2


class PlenoxelLidarSensor(RawSensor):
    """ Lidar sensor in free roam mode """

    def __init__(
            self,
            grid_ckpt_path,
            grid_cfg_path,
            dset_path,
            num_points=11070,
            #num_points=3200,
            lidar_range=5000,
            #upper_fov=np.pi/4,
            #lower_fov=-np.pi/4,
            upper_fov=30 * np.pi / 180,
            lower_fov=-30 * np.pi / 180,
            num_channels=45,
            #num_channels=32,
            sensor_offset=1.6,
            device="cuda:0",
            include_depth=True,
            use_kernel=True):
        super(PlenoxelLidarSensor,
              self).__init__(grid_ckpt_path,
                             (num_points // num_channels, num_channels), (3, ))

        # Load plenoxel model
        print("Grid checkpoint path:", grid_ckpt_path)
        self._world = svox2.SparseGrid.load(grid_ckpt_path, device=device)
        print('Model:', self._world)
        # Load plenoxel config
        parser = argparse.ArgumentParser()
        config_util.define_common_args(parser, data_req=False)
        args = parser.parse_args([])
        args.config = grid_cfg_path
        config_util.maybe_merge_config_file(args, allow_invalid=True)
        config_util.setup_render_opts(self._world.opt, args)
        dset = datasets[args.dataset_type](
            dset_path, split="test", **config_util.build_data_options(args))
        self.plenoxel_transform_T = dset.transform_T
        self.plenoxel_transform_scale = dset.transform_scale

        self._upper_fov = upper_fov
        self._lower_fov = lower_fov
        #self._range = lidar_range
        #self._num_points = num_points
        #self._num_channels = num_channels
        #self._sensor_offset = sensor_offset
        self.num_points = num_points
        self.height = num_channels
        self.width = num_points // num_channels
        self.include_depth = include_depth
        self.use_kernel = use_kernel

    def get_rays(self, c2w):
        rays_o, rays_d = get_lidar_rays(c2w, self.width, self.height,
                                        self._lower_fov, self._upper_fov, old_transform=True)
        return svox2.Rays(rays_o, rays_d)

    def yaw_to_mat(self, yaws):
        # figure out what this matrix does
        Rp = torch.tensor([[0., 0., 1.,],
                           [1., 0., 0.],
                           [0., -1., 0.]]).type_as(yaws)
        return yaw_to_mat(yaws, Rp, y_down=False)

    def read(self, state, cube_params):

        loc = torch.cat([state[:2].cpu(), torch.zeros(1)])
        rot = self.yaw_to_mat(state[2].cpu())
        if cube_params is not None:
            cube_params = cube_params.view(-1, 12)
            cube_loc = cube_params[:, :3]
            cube_props = cube_params[:, 3:]

        mat = torch.zeros(4, 4, dtype=torch.float32)
        mat[:3, :3] = rot
        mat[:3, 3] = loc
        mat[3, 3] = 1.

        # Transformation that was done on carla dataset (excluding z)
        mat[:2, 3] = normalize_pose_xy(mat[:2, 3])

        # Tranformation done by plenoxels
        mat = torch.from_numpy(self.plenoxel_transform_T).float() @ mat
        mat[:3, 3] *= self.plenoxel_transform_scale

        mat = mat.to(state.device)

        rays = self.get_rays(mat)
        #print(self._world.center, self._world.radius)
        im = self._world.volume_render(rays, use_kernel=self.use_kernel).view(
            self.height, self.width, 3)

        #with torch.enable_grad():
        #    print(torch.autograd.grad(im, rays.origins)[0])
        #    print(torch.autograd.grad(im, rays.dirs)[0])
        #    exit()

        if self.include_depth:
            depth = self._world.volume_render_depth(rays).view(
                self.height, self.width)

        debug = False
        if debug:
            print('Output shape:', im.shape)
            im_show = im.cpu().detach().numpy()
            im_show = (im_show * 255).astype(np.uint8)
            imageio.imwrite('tmp_rgb_out.png',im_show)
            if self.include_depth:
                depth_show = depth.cpu().detach().numpy()
                depth_show = (depth_show * 255).astype(np.uint8)
                imageio.imwrite('tmp_depth_out.png',depth_show)

        if self.include_depth:
            output = torch.cat([depth.unsqueeze(-1), im], dim=-1)
            return output.transpose(0, 1)
        else:
            return im.transpose(0, 1)
