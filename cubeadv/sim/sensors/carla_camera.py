from concurrent.futures import ProcessPoolExecutor
import torch
from collections import deque
import threading
import time

from cubeadv.fields.utils import normalize_ngp
import cubeadv.utils as util

import carla

from cubeadv.sim.sensors.sensor_utils import get_camera_rays, get_ndc_ray_grid, transform_rays_to_camera
from cubeadv.fields.utils import transform_ray_to_object
from .raw_sensor import Sensor

from typing import Union, Tuple, List
from math import atan2, degrees

import numpy as np

class CarlaCamera:
    def __init__(self,
                 carla_world : carla.World,
                 width : int=640,
                 height : int=320,
                 focal_length_px : int=320,
                 base_offset : Union[Tuple, torch.Tensor]=[0., 0., 2.],
                 pixel_center : Union[Tuple, torch.Tensor]=(0., 0.),
                 far_plane : float=10.,
                 base_rotation : float=0.):

        self.world = carla_world
        fov = degrees(atan2(width, 2*focal_length_px))*2

        self.camera_rgb = self.spawn_camera(carla_world, 'sensor.camera.rgb', width, height, fov)
        self.camera_depth = self.spawn_camera(carla_world, 'sensor.camera.depth', width, height, fov)

        self.cond = threading.Condition()
        self.depth_cond = threading.Condition()
        self.rgb_queue = deque(maxlen=10)
        self.depth_queue = deque(maxlen=10)

        self.camera_rgb.listen(lambda x: self.on_data(x, self.rgb_queue, self.cond))
        self.camera_depth.listen(lambda x: self.on_data(x, self.depth_queue, self.depth_cond))
        self.base_rotation = base_rotation

        self.width = width
        self.height = height
        self.half_tan_fov = width / (2 * focal_length_px)
        self.base_offset = base_offset
        self.setup_weather(carla_world)

    def spawn_camera(self, carla_world, typestr, width, height, fov):
        camera_bp = carla_world.get_blueprint_library().find(typestr)
        camera_bp.set_attribute('image_size_x', str(width))
        camera_bp.set_attribute('image_size_y', str(height))
        camera_bp.set_attribute('fov', str(fov))

        camera = carla_world.spawn_actor(camera_bp,
                                   carla.Transform(carla.Location(x=0, y=0, z=0), carla.Rotation(yaw=0.)))
        return camera

    def __del__(self):
        self.camera_rgb.stop()
        self.camera_depth.stop()

    def setup_weather(self, world):
        c = world.get_weather()
        c.cloudiness = 0.
        c.precipitation = 0.
        c.precipitation_deposits = 0.
        c.wind_intensity = 0.
        c.sun_azimuth_angle = 250.
        c.fog_density = 0.
        c.sun_altitude_angle = 90.
        c.wetness = 0.
        world.set_weather(c)

    def on_data(self, data, queue, cond):
        with cond:
            queue.append(data)
            cond.notify()

    def get(self, frame, cond, queue):
        with cond:
            item = None
            while not item:
                retries = 0
                while (item is None or item.frame <= frame) and retries < 10:
                    try:
                        item = queue.pop()
                    except:
                        retries += 1
                        time.sleep(0.005)
                        continue
        return item

    def capture(self, tr):
        self.camera_rgb.set_transform(tr)
        self.camera_depth.set_transform(tr)
        _ = self.world.wait_for_tick()
        _ = self.world.wait_for_tick()
        _ = self.world.wait_for_tick()
        frame = self.world.wait_for_tick()
        _ = self.world.wait_for_tick()
        _ = self.world.wait_for_tick()

        rgb = self.get(frame.frame, self.cond, self.rgb_queue)
        depth = self.get(frame.frame, self.depth_cond, self.depth_queue)
        return rgb, depth

    def convert_to_carla_pose(self, state : torch.Tensor, height : float):
        state = state.detach().cpu().numpy()
        l = carla.Location(x=float(state[0]) + self.base_offset[0], y=float(state[1]) + self.base_offset[1], z=self.base_offset[2])
        r = carla.Rotation(yaw=degrees(float(state[2])) + self.base_rotation)
        return carla.Transform(location=l, rotation=r)

    def convert_to_array(self, data, dim=4):
        array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (data.height, data.width, dim))[:,:,:3]
       # array = array.astype(np.float32)
        rgb = array[:,:,::-1]
        return torch.from_numpy(np.ascontiguousarray(rgb))


    def convert(self, rgb, depth):
        rgb = self.convert_to_array(rgb)
        depth = self.convert_to_array(depth).float()
        depth = (depth[:, :, 0] +
                 depth[:, :, 1] * 256 +
                 depth[:, :, 2]*256*256) / (256*256*256 - 1)
        depth = depth * 1000

        return rgb, depth

    def read_internal(self, loc, rot, depth_alignment_factor=60.):
        carla_pose = carla.Transform(loc, rot)
        image, depth = self.capture(carla_pose)
        image, depth = self.convert(image, depth)

        dirs = get_ndc_ray_grid(self.width, self.height,
                                      self.half_tan_fov)
        dirs_im = dirs.view(self.height, self.width, 3)
        depth = depth / dirs_im[:, :, 0].abs()
        depth = depth / depth_alignment_factor
      #  depth = depth * dirs_im[:, :, 0].abs()

        image = image.float() / 255.
        return image, depth


    def read(self, state : torch.Tensor, height=2.8, convert: bool=False, depth_alignment_factor=60.) -> torch.Tensor:
        if state.dim() == 1:
            state = [state]

        images = []
        depths = []
        for state_ in state:
            carla_pose = self.convert_to_carla_pose(state_, height)
            image, depth = self.capture(carla_pose)
            image, depth = self.convert(image, depth)

            dirs = get_ndc_ray_grid(self.width, self.height,
                                          self.half_tan_fov)
            dirs_im = dirs.view(self.height, self.width, 3)
            depth = depth / dirs_im[:, :, 0].abs()
            depth = depth / depth_alignment_factor
          #  depth = depth * dirs_im[:, :, 0].abs()

            if convert:
                image = image.float() / 255.
            images.append(image)
            depths.append(depth)

        return torch.stack(images), torch.stack(depths)

class CarlaCameraCompose(CarlaCamera):
    def __init__(self, *args, box=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_fields = []
        self.scene_extent = 0.25
        if box:
            self.object_aabb = 4.
            self.prescale = 1.
        else:
            self.object_aabb = 4.
            self.prescale = 1.5
        self.box = box
        self.scene_to_object_ratio = (self.object_aabb / self.scene_extent) * self.prescale

        self.nerf_aabb = 3.

        if not box:
            self.scene_prescale = 20. * self.nerf_aabb
        else:
            self.scene_prescale = 30. * self.nerf_aabb


    def add_obj_field(self, field, location=None):
        if location is None:
            location = torch.zeros(4).cuda()
        self.obj_fields.append((field, location.cuda()))

    def add_object(self, field, location):
        self.obj_fields.append((field, location.cuda()))

    @torch.no_grad()
    def read(self, state : torch.Tensor, height=2.8):
        if state.dim() == 1:
            state = state.unsqueeze(0) # batch dim
        rgb_scene, depth_scene = super().read(state, height, convert=True, depth_alignment_factor=self.scene_prescale)

        dirs = get_ndc_ray_grid(self.width, self.height,
                                      self.half_tan_fov)

        rgb_scene = rgb_scene.cuda()
        depth_scene = depth_scene.cuda()

        origins, dirs = transform_rays_to_camera(dirs, state, self.base_rotation)

       # origins = normalize_ngp(origins, self.nerf_aabb, True)
        origins = normalize_ngp(origins, util.OLD_CARLA_MIDPOINT.cuda(), util.OLD_CARLA_SCALE.cuda())
        for field, loc in self.obj_fields:
            # See NGPComposeField for the magic values.
            origins_, dirs_ = transform_ray_to_object(origins.cuda(), dirs.cuda(), loc[:3], loc[3], 16.)
            rescaled_origins = normalize_ngp(origins_, -field.mid, 1. / field.scale)
            rb = field._render(["rgb", "depth", "alpha"], rescaled_origins, dirs_, max_t=10.)
            rgb = rb.rgb.view(state.shape[0], self.height, self.width, -1)
            depth = rb.depth.view(state.shape[0], self.height, self.width)
            hit = rb.hit.view(state.shape[0], self.height, self.width)
            depth = depth / self.scene_to_object_ratio

            alpha = rb.alpha.view(state.shape[0], self.height, self.width, 1)
            alpha_thresh = alpha > 0.95
            alpha_thresh = alpha_thresh[:, :, :, -1]

            object_mask = (depth <= depth_scene) * hit * alpha_thresh

            rgb_scene[object_mask] = rgb[object_mask]
            depth_scene[object_mask] = depth[object_mask]


        return rgb_scene, depth_scene

class BatchCarlaCamera:
    def __init__(self,
                 worlds : List[carla.World],
                 width : int=640,
                 height : int=320,
                 focal_length_px : int=320,
                 pixel_center : Union[Tuple, torch.Tensor]=(0., 0.),
                 far_plane : float=10.,
                 base_rotation : float=0.):
        self.cameras = [CarlaCamera(world, width, height, focal_length_px, pixel_center, far_plane, base_rotation) for world in worlds]
        self.pool = ProcessPoolExecutor(self.cameras.count)

    def read_camera(self, camera, state):
        return camera.read(state.view(1, -1))

    def read(self, states : torch.Tensor):
        images = [i for i in self.pool.map(self.read_camera, self.cameras, tuple(states))]
        return torch.cat(images, dim=0)