import torch
import numpy as np


def yaw_to_mat(yaws, premul, y_down=True):
    # Treats y down so positiv
    yaws = yaws.view(-1, 1)
    if y_down:
        yaws = -yaws

    cos = yaws.cos()
    sin = yaws.sin()

    K = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 0.]],
                     device=yaws.device)

    # cross product matrix for crossing with the [0, 1, 0] vector
    K = torch.tensor([[0., 0., 1.], [0., 0., 0.], [-1., 0., 0.]]).type_as(yaws)
    KK = K.mm(K)
    KK = KK.expand(yaws.shape[0], -1, -1)
    K = K.expand(yaws.shape[0], -1, -1)

    I = torch.eye(3, device=yaws.device).expand(yaws.shape[0], -1, -1)

    R = I + sin.view(-1, 1, 1) * K + (1 - cos).view(-1, 1, 1) * KK
    Rf = torch.matmul(premul, R)
    return Rf


def get_lidar_rays(c2w, width, height, fov_lower, fov_upper, old_transform=False):
    phi = np.linspace(-np.pi, np.pi, width).astype(dtype=np.float32)
    theta = np.linspace(fov_lower, fov_upper, height).astype(dtype=np.float32)

    pv, tv = np.meshgrid(phi, theta)
    forward = np.sin(pv) * np.cos(tv)
    right = np.cos(pv) * np.cos(tv)
    up = np.sin(tv)

    x = forward
    y = -up
    z = -right
    if old_transform:
        x = forward
        y = up
        z = right

    rays_d = torch.from_numpy(np.stack([x, y, z]).reshape(3,
                                                          -1)).to(c2w.device)
    rays_d = c2w[:3, :3].matmul(rays_d).T.contiguous()

    rays_o = c2w[:3, 3].reshape((1, 3))
    rays_o = rays_o.expand(rays_d.shape[0], -1).contiguous()

    return rays_o, rays_d