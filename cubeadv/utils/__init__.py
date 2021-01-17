import numpy as np
import torch
from .torch_utils import *

CUBE_PARAM_SIZE = 12

OLD_CARLA_MIDPOINT = torch.tensor([92.5, 132.5, 0.])
OLD_CARLA_SCALE = torch.tensor([90., -90., 1.])

NEW_CARLA_MIDPOINT = torch.tensor([94., 125., 0.])
NEW_CARLA_SCALE = torch.tensor([16., -15., 1.])

MY580_MIDPOINT = torch.tensor([0., 0., 0.])
MY580_SCALE = torch.tensor([1., 1., 1.])

def rescale(x, mid, scale):
    return (x - mid) / scale
    
def get_permutation(perm , homog=False):
    """
    Get a matrix pertaining to a permutation of the columns.
    Args:
        perm: tensor with the column perms on the last dimension.
        homog: will append an extra dimension to he output with 1 on the diagonal
               padded zeros around
    """
    e = torch.eye(perm.shape[-1])
    e = e[perm].transpose(-1, -2)
    out = e
    if homog:
        out = torch.eye(perm.shape[-1] + 1)
        out = out.broadcast_to(e.shape[:-2] + out.shape[-2:])
        out[..., :-1, :-1] = e
    return out

def get_nerf_max():
    return torch.from_numpy(np.array([100., 140.]))

def get_nerf_min():
    return torch.from_numpy(np.array([85., 125.]))

def normalize_pose_xy(pos_xy):
    max = get_nerf_max().type_as(pos_xy)
    min = get_nerf_min().type_as(pos_xy)
    out = pos_xy.clone()
    out = ((out - min) / (max - min) - 0.5) * 0.5
    return out

def normalize_instant_ngp(pos, aabb_scale):
    # The mapping used to normalize our datasets
    max = torch.tensor([110., 110.]).type_as(pos)
    min = torch.tensor([80., 140.]).type_as(pos)
    
    scale = torch.tensor([16., -15.]).type_as(pos)
    offset =  torch.tensor([94., 125.]).type_as(pos)
    out = pos.clone()
 #   out = ((out - min) / (max - min) - 0.5) * 0.5
    out = (out - offset) / scale

    # divide by the scale to replicate what wisp is doing
    return out / aabb_scale
    
def normalize(x, max, min):
    out = x.clone()
    out = out.view(-1, CUBE_PARAM_SIZE)
    out[:, :2]  = ((out[:, :2] - min) / (max - min) - 0.5) * 0.5
    out[:, 2] = (out[:, 2] - 2.8) / 30.
    out[:, 6:9] = out[:, 6:9] / 30.
    out = out.view(-1)
    return out

def denormalize(params, max, min):
    params = params.clone()

    params = params.view(-1, CUBE_PARAM_SIZE)
    params[:, :2] = (params[:, :2]  / 0.5 + 0.5) *  (max - min) + min
    params[:, 2] = params[:, 2] * 30.+2.8

    # box_size
    params[:,6:9] =  params[:,6:9] * 30.
    params = params.view(-1)
    return params
