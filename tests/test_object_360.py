import torch
import numpy as np

from test_utils import driving_test, image_collection_test

from cubeadv.fields import NGPField
from cubeadv.sim.sensors import Camera

from cubeadv.sim.sensors.sensor_utils import yaw_to_mat


from cubeadv.utils import make_functional, set_weights, get_nerf_max, get_nerf_min

@image_collection_test
def test_object_multiview(request):
    parameter_file = "../exp-params/right-2c-3h/seed-0.npy"
    parameters = torch.from_numpy(np.load(parameter_file))
    
    field = NGPField("wisp/configs/ngp_hydrant_new.yaml")
    
    param_filter = lambda name: 'codebookcolor' in name
    p_ = make_functional(field.pipeline.nef, param_filter=param_filter, verbose=True)
    current_params = p_.param_vector
    p = parameters[:current_params.shape[0]:].cuda()
   
    print(current_params.shape)
    set_weights(field.pipeline.nef, p_, 
                current_params)# + p)
    
    with torch.no_grad():
        camera = Camera(1280, 720, 640)
        render_fn = lambda x: camera.read(field, x)
        center = (get_nerf_max() + get_nerf_min()) / 2
        radius = 90.
        
        samples = torch.linspace(-torch.pi, torch.pi, 6)
        dirs = torch.stack([samples.cos(), -samples.sin()], dim=-1)
        
        points = center + dirs * radius
        angles = ((torch.pi - samples) + torch.pi) % (2*torch.pi) - torch.pi
        mat = yaw_to_mat(angles)
        images = []
        
        for point, angle in zip(points, angles):
            angle = -angle
        #    angle -= 0.5
            render_c2w  = torch.tensor([[1, 0., 0,  92.5],
                                       [0, 0.,  1.,  302.5],
                                       [0., -1, 0.,   0.],
                                       [0., 0., 0., 0.]]).cuda()
            offset_ = point.cuda() - torch.tensor([92.5, 132.5]).cuda()
         #   offset_ = torch.tensor([-100., 1.])
            offset = torch.zeros(3).cuda()
            offset[0] = offset_[0]
            offset[2] = offset_[1]
            mat = torch.tensor([[angle.cos(), 0, -angle.sin()], [0, 1, 0], [angle.sin(), 0, angle.cos()]]).cuda()
            
            angle2 = torch.tensor(0.1)
            mat2 = torch.tensor([[angle2.cos(), -angle2.sin(), 0], [angle2.sin(), angle2.cos(), 0], [0., 0., 1]]).cuda()
            offset_W = mat.matmul(offset)
            
            render_c2w[0, 3] -= 70.
            render_c2w[1, 3] = 142.5
            render_c2w[2, 3] = -0.15
            
           # mat=torch.eye(3).cuda()
            render_c2w[:3, :3] = mat2.matmul(mat.matmul(render_c2w[:3, :3]))
            
            rb = camera.read_internal(field, render_c2w)
            images.append(rb.rgb.reshape(720, 1280, 3).detach().cpu())
            
        states = torch.cat([points, angles.unsqueeze(-1)], dim=-1).cuda().float()
        print(states)
      #  images, depths = render_fn(states)
      #  images = images.detach().cpu()
        
    return None, None, images, "0"
    