import time
import copy

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from .raw_sensor import RawSensor
from ..simple_sim import UNet


class FreeLidarSensor(RawSensor):
    """ Lidar sensor in free roam mode """

    def __init__(self,
                 sim_builder,
                 num_points=3200,
                 lidar_range=5000,
                 upper_fov=10,
                 lower_fov=-30,
                 num_channels=32,
                 sensor_offset=1.6
                 ):
        super(FreeLidarSensor, self).__init__(
            sim_builder, (num_points // num_channels, num_channels), (3, ))


        self._world = sim_builder(16 + 12, num_channels, num_points // num_channels)
        self._upper_fov = upper_fov
        self._lower_fov = lower_fov
        self._range = lidar_range
        self._num_points = num_points
        self._num_channels = num_channels
        self._sensor_offset = sensor_offset

    def set_params(self, models_new):
        self._world.models.update(models_new)
    
    def get_params(self):
        return self._world.models

    def yaw_to_mat(self, yaws):
        Rp = torch.tensor([[0., 0., 1.,], [1., 0., 0.], [0., -1., 0.]]).type_as(yaws)
        yaws = yaws.view(-1, 1)

        cos = yaws.cos()
        sin = yaws.sin()

        K = torch.tensor([[0., -1., 0.],
                          [1., 0., 0.],
                          [0., 0., 0.]], device=yaws.device)

        K = torch.tensor([[0., 0., 1.], [0., 0., 0.], [-1., 0., 0.]]).type_as(yaws)
        KK = K.mm(K)
        KK = KK.expand(yaws.shape[0], -1, -1)
        K = K.expand(yaws.shape[0], -1, -1)

        I = torch.eye(3, device=yaws.device).expand(yaws.shape[0], -1,  -1)

        R = I +  sin.view(-1, 1, 1) * K + (1 - cos).view(-1, 1, 1) * KK
        Rf = torch.matmul(Rp, R)

        return Rf


    def read(self, state, cube_params):
        sensor_state = state

        loc = torch.cat([state[:2], torch.zeros(1, device=state.device)])
        rot = self.yaw_to_mat(state[2])
        cube_params = cube_params.view(-1, 12)
        cube_loc = cube_params[:, :3]
        cube_props = cube_params[:, 3:]

        mat = torch.zeros(4, 4, device=state.device, dtype=torch.float32)
        mat[:3, :3] = rot
        mat[:3, 3] = loc
        mat[3, 3] = 1.

        im, depth = self._world.run(mat, cube_loc, cube_props, rgb=True)
        if torch.any(torch.isnan(depth)):
            print("Nan found at: ", state)

        output = torch.cat([depth.unsqueeze(-1), im], dim=-1)
        return output.transpose(0, 1)

    def read_batch(self, state, cube_loc):
        sensor_state = state

        loc = torch.cat([state[:, :2], torch.zeros(state.shape[0], 1, device=state.device)], dim=1)
        rot = self.yaw_to_mat(state[:, 2])
        cube_location = torch.cat([cube_loc, torch.zeros(state.shape[0], 1, device=state.device)], dim=1)

        rot = rot[:, :, [1, 2, 0]]
        rot[:, :, 1] = -rot[:, :, 1]

        mat = torch.zeros(state.shape[0], 4, 4, device=state.device)
        mat[:, :3, :3] = rot
        mat[:, :2, 3] = loc[:, :2]
        mat[:, 3, 3] = 1.

        depth = self._world.query_batch(mat, cube_location)
        if torch.any(torch.isnan(depth)):
            pass

        return depth.to(state.device)

    def get_sim(self):
        return self._world


    def get_grad(self, state, cube_loc, z=1.7):
        raise NotImplementedError

    def get_cube_grad(self, state, cube_loc, z=2.7):
        raise NotImplementedError

