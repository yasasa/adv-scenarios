from typing import Union, Tuple

import torch
import numpy as np
from cubeadv import utils

from cubeadv.fields.base_rf import BaseRadianceField
from cubeadv.fields.utils import normalize_ngp
from .raw_sensor import Sensor
from .sensor_utils import get_camera_rays, get_ndc_ray_grid

class OrthographicCamera(Sensor):
    def __init__(self,
                 width : int=640,
                 height : int=320,
                 sensor_width : float= 0.5,
                 pixel_center : Union[tuple, torch.Tensor]=(0., 0.),
                 far_plane : float=10.,
                 base_rotation : float=0.):

        self.width = width
        self.height = height
        self.pixel_center = pixel_center
        self.sensor_width = sensor_width
        self.camera_far = far_plane
        self.base_rotation = base_rotation

    def get_ortho_ray_grid(self, width, height, sensor_width):
        sensor_height = height * sensor_width / width
        xs = torch.linspace(-sensor_width/2, sensor_width/2, width)
        ys = torch.linspace(-sensor_height/2, sensor_height/2, height)
        px, py = torch.meshgrid(xs, ys, indexing='xy')

        pxyz = torch.stack([
                torch.ones_like(px),
                torch.zeros_like(px),
                torch.zeros_like(px)
             ], dim=-1).view(-1, 3)

        offsets = torch.stack([torch.zeros_like(px), -px, py], dim=-1).view(-1, 3)

        return pxyz, offsets



    def read_internal(self, nerf : BaseRadianceField , c2w : torch.Tensor):
        rays, origin_offsets = self.get_ortho_ray_grid(self.width, self.height, self.sensor_width)
        rays = rays.type_as(c2w)
        origin_offsets = origin_offsets.type_as(c2w)

        rays_w = rays.matmul(c2w[:3, :3].T)
        origin_offsets_w = origin_offsets.matmul(c2w[:3, :3].T)

        origins =  origin_offsets_w + c2w[:3, 3]
        origins = normalize_ngp(origins, utils.OLD_CARLA_MIDPOINT.cuda(), utils.OLD_CARLA_SCALE.cuda())
        origins[:, 2] /= 90
        rb =  nerf._render(["rgb"], origins, rays_w, max_t=self.camera_far, normalize_origins=False)
        rb.rgb = rb.rgb.view(-1, self.height, self.width, 3)
        rb.alpha = rb.alpha.view(-1, self.height, self.width)

        return rb

    def read(self, nerf : BaseRadianceField , state : torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

class Camera(Sensor):
    def __init__(self,
                 width : int=640,
                 height : int=320,
                 focal_length_px : np.typing.ArrayLike = 320,
                 pixel_center : Union[tuple, torch.Tensor]=(0., 0.),
                 far_plane : float=10.,
                 base_rotation : float=0.):

        self.width = width
        self.height = height
        self.pixel_center = pixel_center
        print(focal_length_px)
        self.half_tan_fov = np.array([width, height]) / (2 * np.asarray(focal_length_px))
        self.camera_far = far_plane
        self.base_rotation = base_rotation

    def get_rays(self, state : torch.Tensor):
        if state.dim() <= 1:
            state = state.unsqueeze(0) # Batch dim

        return get_camera_rays(state, self.width, self.height,
                                self.half_tan_fov, self.base_rotation)

    def read(self, nerf : BaseRadianceField , state : torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        origins, rays = self.get_rays(state)
        rgb, depth =  nerf.query(origins, rays, max_t=self.camera_far, **kwargs)
        rgb = rgb.view(-1, self.height, self.width, 3)
        depth = depth.view(-1, self.height, self.width)

        return rgb, depth

    def read_internal(self, nerf : BaseRadianceField , c2w : torch.Tensor):
        rays = get_ndc_ray_grid(self.width, self.height, self.half_tan_fov).type_as(c2w)
        rays_w = rays.matmul(c2w[:3, :3].T)
        origins = c2w[:3, 3].expand_as(rays_w)
        print(origins)
        rb =  nerf._render(["rgb"], origins, rays_w, max_t=self.camera_far)
        rb.rgb = rb.rgb.view(-1, self.height, self.width, 3)
        rb.alpha = rb.alpha.view(-1, self.height, self.width)

        return rb



class CameraRig(Camera):
    def __init__(self, camera_offsets : torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.offsets = camera_offsets

    def read(self, nerf : BaseRadianceField, state: torch.Tensor):
        # have to broadcast the offsets to state and send it in.
        return super().read(nerf, state + self.offsets)