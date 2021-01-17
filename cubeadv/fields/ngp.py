import numpy as np

import torch
from torch.utils.checkpoint import checkpoint

from wisp.core import Rays

from wisp.framework import WispState
import wisp.config_parser as config_parser
from wisp.core.render_buffer import RenderBuffer

from .base_rf import BaseRadianceField
from .utils import normalize_ngp, transform_ray_to_object
import cubeadv.utils as util

from typing import Union

CoordinateBroadcastableType = Union[torch.tensor, list, np.array, float]


def get_forward(T):
    v = torch.tensor([1., 0., 0., 0.]).type_as(T)
    return T @ v

def normalize_alpha(t):
    if t.numel() == 0:
        return t

    min = t.min(dim=0)[0]
    max = t.max(dim=0)[0]
    scale = (max - min).clamp(min=1e-4)

    t = torch.where(scale > 1e-4, (t - min)/scale, t)

    return t

class NGPField(BaseRadianceField, torch.nn.Module):

    def __init__(self, cfg : str,
                 scene_scale : CoordinateBroadcastableType=util.OLD_CARLA_SCALE,
                 scene_midpoint:CoordinateBroadcastableType=util.OLD_CARLA_MIDPOINT,
                 transform : Union[None, torch.tensor]=None):

        torch.nn.Module.__init__(self)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scale = scene_scale.to(self.device)
        self.mid = scene_midpoint.to(self.device)

        if transform is not None:
            self._transform = transform.to(self.device)
        else:
            self._transform = None

        self.reload(cfg)

    def reload(self, cfg: str):
        parser = config_parser.parse_options()
        args = parser.parse_args("")
        args.config = cfg
        configs = config_parser.parse_args(parser, args)
        assert configs.pretrained is not None
        self.render_batch_size = configs.render_batch

        self.pipeline = config_parser.load_neural_pipeline(args=configs, dataset=None, device=self.device)

    def _render(self, channels, origins, dirs, max_t=1., normalize_origins=True):
        if normalize_origins:
            origins = normalize_ngp(origins, self.mid, self.scale)

        if self._transform is not None:
            origins = origins @ self._transform[:3, :3].T + self._transform[:3, -1]
            dirs = dirs @ self._transform[:3, :3].T

        rays = Rays(origins, dirs, dist_max=max_t)
        rb = RenderBuffer()
        if self.render_batch_size > 0:
            for rayp in rays.split(self.render_batch_size):
                if self.render_batch_size > 4000:
                    rb += self.pipeline(channels=channels, rays=rayp)
                else:
                    rb += checkpoint(lambda c, r: self.pipeline(channels=c, rays=r), channels, rayp)
        else:
            rb = self.pipeline(channels=channels, rays=rays)

        return rb

    def query(self, origins, rays, max_t=1., **kwargs):
        rb = self._render(["rgb", "depth"], origins, rays, max_t=max_t, **kwargs)
        return rb.rgb, rb.depth

    def rgb(self, origins, rays, max_t=1.):
        return self._render(["rgb"], origins, rays, max_t=max_t).rgb

    def depth(self, origins, rays, max_t=1.):
        return self._render(["depth"], origins, rays, max_t=max_t).depth


class NGPComposeField(NGPField, torch.nn.Module):

    def __init__(self,
                scene_field : NGPField,
                obj_cfg : str='',
                num_obj : int=1,
                separate_obj_params : bool=False,
                object_scale : CoordinateBroadcastableType = 16.):
        torch.nn.Module.__init__(self)

        self.object_scale = object_scale
        self.depth_scale = self.object_scale

        # Wisp state
        self.scene_state = WispState()

        # Load scene model
        self.scene_field = scene_field
        self.render_batch_size = self.scene_field.render_batch_size

        # Load obj models
        self.obj_fields = []
        if obj_cfg != '':
            for i in range(num_obj):
                if separate_obj_params or i == 0:
                    obj_field = NGPField(obj_cfg)
                   # print("Loaded object-{} model".format(i))
                self.obj_fields.append(obj_field)
        self.obj_param_len = 4

        self.obj_params = torch.zeros(self.obj_param_len*len(self.obj_fields)).cuda()

    def add_obj_field(self, field, loc=None):
        self.obj_fields.append(field)
        if loc is None:
            loc = torch.zeros(self.obj_param_len).type_as(self.obj_params)

        self.obj_params = torch.cat([self.obj_params, loc])

    def set_transform_params(self, transform_params):
        assert(transform_params.shape[0] % self.obj_param_len == 0)
        self.obj_params = transform_params

    def _render(self, channels, origins, dirs, max_t=1., obj_transform=None):
        rays = Rays(origins, dirs, dist_max=max_t)
        rb = RenderBuffer()
        if self.render_batch_size > 0:
            for rayp in rays.split(self.render_batch_size):
                rb += self.partial_render(channels, rayp.origins, rayp.dirs, max_t, obj_transform)
        else:
            rb = self.partial_render(channels, origins, dirs, max_t, obj_transform)

        return rb

    def partial_render(self, channels, origins, dirs, max_t=1., obj_transform=None):
        if obj_transform is None:
            obj_transform = self.obj_params

        rb_scene = self.scene_field._render(channels, origins, dirs, max_t=max_t)
        rb_scene.alpha[rb_scene.hit] = normalize_alpha(rb_scene.alpha[rb_scene.hit])

        # Normalize obj ray origins prior to transformations
        origins = normalize_ngp(origins, self.scene_field.mid, self.scene_field.scale)
        params = obj_transform.split(self.obj_param_len)

        for i in range(len(self.obj_fields)):
            obj_field = self.obj_fields[i]
            tr =  params[i]
            origins_o, dirs_o = transform_ray_to_object(origins, dirs,
                                                        tr[:3], tr[3],
                                                        self.object_scale)

           # new_mid, new_scale = transform_ray_to_object(self.scene_field.mid[None], self.scene_field.scale[None], tr[:3], tr[3], self.object_scale)

            rescaled_origins = normalize_ngp(origins_o, -obj_field.mid, 1. / obj_field.scale)
            rb_obj = obj_field._render(channels, rescaled_origins, dirs_o, max_t=max_t*1.2)
            depth_scale = self.scene_field.scale.norm() / obj_field.scale.norm()
            rb_obj.depth = rb_obj.depth / depth_scale

            # Compose
            # Normalize alphas
            rb_obj.alpha[rb_obj.hit] = normalize_alpha(rb_obj.alpha.clone()[rb_obj.hit])
            # Mask out everything other than the solid object in the object nerf,
            # Need to reshape here for sizes to maek sense in the rb_scene.blend fn
        #    rb_obj.hit = rb_obj.alpha > 0.95
            rb_obj.hit = rb_obj.hit.view(-1, 1)
            rb_scene.hit = rb_scene.hit[:, None]
            # Blend
            rb_scene = rb_scene.blend(rb_obj, channel_kit=self.scene_state.graph.channels)
            rb_scene.hit = rb_scene.hit.squeeze()

        return rb_scene
