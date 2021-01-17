import torch
import numpy as np

from cubeadv.fields.base_rf import BaseRadianceField
from .raw_sensor import Sensor
from .sensor_utils import get_lidar_rays, yaw_to_mat


class Lidar(Sensor):
    def __init__(self,
                 horizontal_beams : int=100,
                 vertical_beams : int=32,
                 max_range : float = 5,
                 max_angle=np.pi/6.,
                 min_angle=-np.pi/6.,
                 base_rotation=0):
                 
        self._upper_fov = max_angle
        self._lower_fov = min_angle
        self.height =  vertical_beams
        self.width = horizontal_beams
        self.max_range = max_range
        self.base_rotation = base_rotation

    def read(self, nerf : BaseRadianceField, state : torch.Tensor):
        if state.dim() <= 1:
            state = state.unsqueeze(0) # Batch dim
    
        batches = state.shape[0]
        
        loc = torch.cat([state[:, :2], 
                         torch.zeros_like(state[:, :1])], dim=-1)
        rot = yaw_to_mat(state[:, 2] + self.base_rotation)


        rays = get_lidar_rays(self.width, self.height,
                              self._lower_fov, self._upper_fov)
                                        
        origins = loc.view(-1, 1, 3).expand(-1, rays.shape[0], -1)
        origins = origins.type_as(state).contiguous().view(-1, 3)
        
        rays = rays.view(1, -1, 3).type_as(state)
        rays = torch.matmul(rays, rot.mT).contiguous().view(-1, 3)
                                        
        rgb, depth =  nerf.query(origins, rays, max_t=self.max_range)
        
        rgb = rgb.view(batches, self.height, self.width, 3)
        depth = depth.view(batches, self.height, self.width)

        return rgb, depth

