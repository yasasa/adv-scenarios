import time
import copy

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from .raw_sensor import RawSensor
from ..simple_sim import UNet
from cubeadv.nerf import make_sim

class FreeCameraSensor(RawSensor):
    def __init__(self,
                 sim_builder,
                 frame_height=360,
                 frame_width=640,
                 fov=90,
                 cx=0.5,
                 cy=0.5):
        super(FreeCameraSensor, self).__init__(
            sim_builder, (frame_width, frame_height), (3, ))


        self._world = sim_builder(16 + 3, frame_height, frame_width)
        self.frame_height = frame_height
        self.frame_width = frame_width

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

        mat = torch.zeros(4, 4, device=state.device)
        mat[:3, :3] = rot
        mat[:3, 3] = loc
        mat[3, 3] = 1.

        rgb, depth = self._world.run(mat, cube_loc, cube_props, rgb=True)

        return rgb.to(state.device)
