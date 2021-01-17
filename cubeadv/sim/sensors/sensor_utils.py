import torch
import numpy as np

from typing import Optional, Union

def yaw_to_mat(yaws : torch.Tensor, premul : Optional[torch.Tensor]=None):
    """
    Generates a rotation matrix from a given yaw angle in the canonical
    coordinate system:

        Z      Y
        ^    /
        |  /
        |---------> X(FORWARD)

    The yaw rotation is applied along the Z axis.

    Args:
        yaws: The yaws to build matrices out of.
        premul: Optional matrix to premultiply by to bring the matrix to a
                desired frame.
    """
    yaws = yaws.view(-1, 1)

    cos = yaws.cos()
    sin = yaws.sin()

    Kz = torch.tensor([[0., -1., 0.],
                        [1., 0., 0.],
                        [0., 0., 0.]],
                        device=yaws.device)
  # Ky = torch.tensor([[0., 0., -1.],
  #                     [0., 0., 0.],
  #                     [1., 0., 0.]],
  #                     device=yaws.device)
  #  Kz = Ky
    KK = Kz.mm(Kz)
    KK = KK.expand(yaws.shape[0], -1, -1)
    Kz = Kz.expand(yaws.shape[0], -1, -1)
    
    I = torch.eye(3).type_as(yaws).expand(yaws.shape[0], -1, -1)

    R = I + sin.view(-1, 1, 1) * Kz + (1 - cos).view(-1, 1, 1) * KK
    if premul is not None:
        R = torch.matmul(premul, R)
    return R

def get_lidar_rays(width : int, height : int,
        fov_lower : Union[float, torch.Tensor],
        fov_upper : Union[float, torch.Tensor],
        old_transform : bool=False):
    """
    Generates a grid of lidar rays of the format [width, height], flattened out.

    The coordinate frame to generate this set of rays is the canonical coordinate
    space explained above.
    
    Outputs are in [height...width, 3] format where x...y means that x is
    unraveled before y.

    Args:
        width: number of rays in the horizontal direction.
        height: number of vertical beams.
        fov_lower: Angule of the bottom most ray in radians.
        fov_upper: Angle of the top most ray in radians.
    """
    phi = torch.linspace(-np.pi, np.pi, width)
    theta = torch.linspace(fov_lower, fov_upper, height)

    pv, tv = torch.meshgrid(phi, theta, indexing='xy')
    forward = np.cos(pv) * np.cos(tv)
    right = np.sin(pv) * np.cos(tv)
    up = np.sin(tv)

    x = forward
    y = right
    z = up

    rays_d = torch.stack([x, y, z], dim=-1).view(-1, 3)

    return rays_d

def get_ndc_ray_grid(width : int, height : int, half_tan_fov: np.typing.ArrayLike):
    """
    Generates a grid of pixels of the format [width, height], flattened out.

    Outputs are in [height...width, 3] format where x...y means that x is
    unraveled before y.

    Args:
        width: frame width.
        height: frame height.
        half_tan_fov: half of the tan(fov_vert)
    """
    if np.isscalar(half_tan_fov):
        
        aspect = height / width
        half_tan_fov = np.array([1., aspect])*half_tan_fov

    xs = torch.arange(width) + 0.5
    ys = torch.arange(height) + 0.5
    px, py = torch.meshgrid(xs, ys, indexing='xy')

    py = 2. * py / height - 1.
    px = 2. * px / width - 1.

    pxyz = torch.stack([
            torch.ones_like(px), 
            -px * half_tan_fov[0],
            py * half_tan_fov[1],
         ], dim=-1).view(-1, 3)
         
    pxyz /= torch.linalg.norm(pxyz, dim=-1, keepdims=True)

    return pxyz.view(-1, 3)
    
def transform_rays_to_camera(dirs: torch.Tensor, state : torch.Tensor, rotation_offset : float):
    batches = state.shape[0]
    rot_mat = yaw_to_mat(-(state[:, 2] + rotation_offset)) # take the negative to go to left handed
    rays = dirs.view(1, -1, 3).type_as(state)

    origins = torch.cat([state[:, :2],
                         0.00*torch.ones_like(state[:, :1])], dim=-1)
    origins = origins.view(batches, 1, 3).expand(-1, rays.shape[1], -1)
    origins = origins.contiguous().view(-1, 3)

    rays = torch.matmul(rays, rot_mat.mT)
    rays = rays.contiguous().view(-1, 3)
    return origins, rays
    
def get_camera_rays(state : torch.Tensor, width : int, height : int, half_tan_fov : np.typing.ArrayLike, rotation_offset : float):
    pixel_rays = get_ndc_ray_grid(width, height, half_tan_fov)
    return transform_rays_to_camera(pixel_rays, state, rotation_offset)