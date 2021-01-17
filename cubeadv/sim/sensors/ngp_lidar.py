import sys
import argparse
import time
import copy
from xml.etree.ElementInclude import include

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from cubeadv.fields.utils import inv_homog_transform

from .raw_sensor import RawSensor
from ..simple_sim import UNet
from .sensor_utils import get_lidar_rays, yaw_to_mat
from cubeadv.utils import normalize_instant_ngp, normalize_pose_xy

from wisp.trainers import BaseTrainer
from wisp.utils import PerfTimer
from wisp.ops.image import write_png, write_exr
from wisp.ops.image.metrics import psnr, lpips, ssim
from wisp.core import Rays

from wisp.trainers import *
from wisp.framework import WispState
from wisp.config_parser import *

from wisp.renderer.core.api import add_to_scene_graph
from kaolin.render.camera import Camera
from wisp.ops.raygen import generate_pinhole_rays, generate_ortho_rays, generate_centered_pixel_coords

class InstantNGPLidar(RawSensor):

    def __init__(self,
                 scene_cfg_path,
                 num_points=3200,
                 lidar_range=5000,
                 upper_fov=30 * np.pi / 180,
                 lower_fov=-30 * np.pi / 180,
                 num_channels=32,
                 sensor_offset=1.6,
                 device="cuda:0",
                 include_depth=True,
                 use_kernel=True,
                 aabb_scale=2.,
                 old_dataset=True,
                 obj_cfg_path=None,
                 num_obj=1,
                 separate_object_params=False):
        super(InstantNGPLidar,
              self).__init__([scene_cfg_path],
                             (num_points // num_channels, num_channels), (3, ))

        parser = parse_options(return_parser=True)

        # Register any newly added user classes before running the config parser
        # Registration ensures the config parser knows about these classes and is able to dynamically create them.
        # from wisp.config_parser import register_class
        # from wisp.models.nefs.nerf_2mlp import SeparatedDecoderNeuralRadianceField
        # register_class(SeparatedDecoderNeuralRadianceField, 'SeparatedDecoderNeuralRadianceField')

        # Wisp state
        self.scene_state = WispState()

        # Load scene model
        configs = parse_yaml_config(scene_cfg_path, parser)
        configs = parser.parse_args("")
        assert configs.pretrained is not None
        self.pipeline_scene, _, _ = get_modules_from_config(configs, init_dataset=False)
        add_to_scene_graph(self.scene_state, 'scene', self.pipeline_scene) # Add pipeline to scene graph
        print("Loaded scene model, nef type:", self.pipeline_scene.nef.get_nef_type())
        
        # Load obj models
        self.pipeline_objs = []
        if obj_cfg_path is not None:
            for i in range(num_obj):
                if separate_object_params or i == 0:
                    configs = parse_yaml_config(obj_cfg_path, parser)
                    configs = parser.parse_args("")
                    assert configs.pretrained is not None
                    pipeline_obj, _, _ = get_modules_from_config(configs, init_dataset=False)
                    add_to_scene_graph(self.scene_state, "object_{}".format(i), pipeline_obj)
                    print("Loaded object-{} model, nef type:".format(i), pipeline_obj.nef.get_nef_type())
                self.pipeline_objs.append(pipeline_obj)

        self._upper_fov = upper_fov
        self._lower_fov = lower_fov
        self.num_points = num_points
        self.height = num_channels
        self.width = num_points // num_channels
        self.include_depth = include_depth
        self.scale = aabb_scale
        self.use_kernel = use_kernel
        self.old_dataset = old_dataset

        self.debug = False
        if self.debug:
            self.height = 80
            self.width = 437

    def get_rays(self, c2w):
        rays_o, rays_d = get_lidar_rays(c2w, self.width, self.height,
                                        self._lower_fov, self._upper_fov)
        return Rays(rays_o.type_as(c2w), rays_d.type_as(c2w), dist_max=6)


    def get_rot_mat(self, yaws):
        # Wisp first converts our format to z up format and then internally
        # changes coordinates to blender which takes it back to our coordinate
        # frame
        if self.old_dataset:
            Rp = torch.tensor([[0., 0., -1.], [-1., 0., 0.], [0., -1., 0.]]).type_as(yaws)
        else:
            Rp = torch.tensor([[0, 0., -1],[0, 1., 0], [1, 0, 0.]]).type_as(yaws)
            #Rp = torch.eye(3).type_as(yaws)
        return yaw_to_mat(-yaws, Rp)

    def read(self, state, obj_params):

        # ========================= Render Scene ===============================

        loc = torch.cat([state[:2], torch.zeros(1).type_as(state)])
        rot = self.get_rot_mat(state[2])

        mat = torch.zeros(4, 4, dtype=torch.float32).to(state.device)
        mat[:3, :3] = rot
        mat[:3, 3] = loc
        mat[3, 3] = 1.

        # convert carla coordinates to normalized coordinates
        if self.old_dataset:
            loc = normalize_pose_xy(mat[:2, 3])
            loc /= self.scale # replicating what wisp is doing
            mat[:2, 3] = loc
            mat[1, 3] *= -1
        else:
            loc = normalize_instant_ngp(mat[:2, 3], self.scale)
            # [xyz] [xzy]
            mat[0, 3] = loc[0]
            mat[1, 3] = mat[2, 3]
            mat[2, 3] = loc[1]

        rays = self.get_rays(mat)

        channels = ['rgb']
        # Note: render depth for object composition even if include_depth = False for policy
        channels.append('depth')
        
        # Render scene
        renderbuffer_scene = self.pipeline_scene(channels=channels, rays=rays)
        renderbuffer_scene = renderbuffer_scene.reshape(self.height, self.width, -1)

        # # ========================= Render Object ==============================

        out_rb = renderbuffer_scene

        # Render object
        for i in range (len(self.pipeline_objs)):
            pipeline_obj = self.pipeline_objs[i]

            # Sample camera matrix from Lego V8 Engine RTMV dataset
            H_c0_o0 = torch.tensor([[0.0420, 0.0000, 0.9991, 0.1021],
                                          [0.8454, 0.5329, -0.0356, 0.5348],
                                          [-0.5324, 0.8462, 0.0224, -3.1045],
                                          [0.0, 0.0, 0.0, 1.0]]).cuda()

            # [Initial camera] to [initial object] matrix for turn trajectory (currently 
            # initial object placement in world is randomly picked using a 
            # sample camera matrix from the object dataset)
            H_o0_c0 = inv_homog_transform(H_c0_o0)

            # [Initial camera] to world matrix for turn trajectory
            H_w_c0 = torch.tensor([[0., 0., 1., 0.1],
                               [1., 0., 0., 0.05],
                               [-0., -1., 0., 0.],
                               [0., 0., 0., 1.]]).cuda()

            # World to [initial object] matrix
            H_o0_w = H_o0_c0 @ inv_homog_transform(H_w_c0)

            # Homogeonous ray origins and directions in world frame
            rays_origins = torch.cat((rays.origins, torch.ones(rays.origins.shape[0], device=rays.origins.device).view(-1,1)), dim=1).view(-1,4,1)
            rays_dirs = rays.dirs.view(-1,3,1)

            # Apply object transformations in world frame
            if obj_params is not None:
                T = torch.eye(4).type_as(H_o0_w)
                T[:3,3] = obj_params[i*3:(i+1)*3]  # translation
                #T[:3,3] *= obj_params  # scale
                inv_T = inv_homog_transform(T)
                rays_origins = inv_T.matmul(rays_origins)
                rays_dirs = inv_T[:3,:3].matmul(rays_dirs)

            # Transform rays to object frame
            origins_o = H_o0_w.matmul(rays_origins)[:,0:3,0]
            dirs_o = H_o0_w[:3, :3].matmul(rays_dirs)[:,:,0]
            # if obj_params is None:  # TODO: REMOVE
            #     origins_o = origins_o.detach()
            #     dirs_o = dirs_o.detach()
            rays_o = Rays(origins_o, dirs_o, dist_max=6)

            renderbuffer_obj = pipeline_obj(channels=channels, rays=rays_o)
            renderbuffer_obj = renderbuffer_obj.reshape(self.height, self.width, -1)
            if self.debug:
                print("Obj rgb shape:", renderbuffer_obj.rgb.shape)
                self.save_img(renderbuffer_obj.rgb, 'tmp_obj_rgb_out.png')

            # ============================ Compose =============================

            renderbuffer_obj.depth /= 20  # TODO: remove

            out_rb = out_rb.blend(renderbuffer_obj, channel_kit=self.scene_state.graph.channels) 

        # =============================== Output ===============================

        rgb = out_rb.rgb
        if self.include_depth:
            depth = out_rb.depth
        
        if self.debug:
            print('Output rgb shape:', rgb.shape)
            self.save_img(rgb, 'tmp_rgb_out.png')
            if self.include_depth:
                self.save_img(depth, 'tmp_depth_out.png')

            # #rgb.requires_grad = True
            # rays.origins.requires_grad = True
            # rays.dirs.requires_grad = True
            # print(rays.origins.shape)
            # print(rays.origins.requires_grad)
            # print(rays.dirs.shape)
            # print(rays.dirs.requires_grad)
            # with torch.enable_grad():
            #    print(torch.autograd.grad(rgb, rays.origins)[0])
            #    print(torch.autograd.grad(rgb, rays.dirs)[0])

        if self.include_depth:
            output = torch.cat([rgb, depth], dim=2)
        else:
            output = rgb

        return output.transpose(0, 1)
