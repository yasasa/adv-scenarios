import torch

import cubeadv.utils as util

def inv_homog_transform(H):
    mat = torch.zeros(4, 4).type_as(H)
    mat[:3, :3] = H[:3,:3].T
    mat[:3, 3] = -H[:3,:3].T @ H[:3,3]
    mat[3, 3] = 1.
    return mat
    
def normalize_ngp(loc, center, scale):
    loc = loc.clone()
    return util.rescale(loc, center, scale)

    return loc

def get_forward(T):
    v = torch.tensor([1., 0., 0., 0.]).type_as(T)
    return T @ v
    
def transform_ray_to_object(origins, dirs, object_loc, 
                            object_rot, object_aabb):
                            
    canonical_rot = torch.eye(3).type_as(origins)
    H_c0_o0 = torch.eye(4).type_as(origins)
    H_c0_o0[:3, :3] = canonical_rot

    # [Initial camera] to [initial object] matrix for turn trajectory (currently
    # initial object placement in world is randomly picked using a
    # sample camera matrix from the object dataset)
    H_o0_c0 = inv_homog_transform(H_c0_o0)

    H_w_c0 = torch.tensor([[1., 0., 0., 0.],
                           [0., 0., 1.,0.0],
                           [0., 1., 0., 0.],
                           [0., 0., 0., 1.]]).type_as(origins)

    H_o0_w = H_o0_c0 @ inv_homog_transform(H_w_c0)
    
    # Homogeonous ray origins and directions in world frame
    rays_origins = torch.cat((origins, torch.ones(origins.shape[0], device=origins.device).view(-1,1)), dim=1).view(-1,4,1)
    rays_dirs = dirs.view(-1,3,1)

    # Apply object transformations in world frame
    if object_loc is not None:
        from cubeadv.sim.sensors.sensor_utils import yaw_to_mat
        T = torch.eye(4).type_as(H_o0_w)
        
        translation =  object_loc
        T[:3,:3] = yaw_to_mat(object_rot.view(1, 1)).squeeze()
        T[:3,3] = translation
        inv_T = inv_homog_transform(T)
        
        rays_origins = inv_T.matmul(rays_origins)
        rays_dirs = inv_T[:3,:3].matmul(rays_dirs)

    rays_origins[:, 0:3] = rays_origins.clone()[:, 0:3] * object_aabb

    # Transform rays to object frame
    origins_o = H_o0_w.matmul(rays_origins)[:,0:3,0]
    dirs_o = H_o0_w[:3, :3].matmul(rays_dirs)[:,:,0]
    dirs_o = dirs_o / dirs_o.norm(dim=-1, keepdim=True)
    
    return origins_o, dirs_o

