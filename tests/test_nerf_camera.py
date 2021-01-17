import os
from pathlib import Path
import time
import signal
from tkinter import W

import pytest
import subprocess

#import carla
import imageio
import torch

from cubeadv.sim.sensors import Camera
from cubeadv.fields import NGPField, NGPComposeField

import cubeadv.utils as util
from cubeadv.utils import make_functional, set_weights
from test_utils import *
import numpy as np

@pytest.mark.parametrize("location", torch.tensor([[100.4, 132.5, np.pi], [88.4, 114.5, np.pi/2]]))
@image_comparison_test
@torch.no_grad()
def test_nerf_camera(location):
    field = NGPComposeField(NGPField("wisp/configs/ngp_nerf_bg_new.yaml"))
    field.scene_field.pipeline.nef.ignore_view_dir = False
                     
    hydrant_field = NGPField('wisp/configs/ngp_hydrant_new.yaml', scene_scale=torch.ones(3), scene_midpoint=torch.zeros(3))
    hydrant_field1 = NGPField('wisp/configs/ngp_hydrant_new.yaml',
    scene_scale=torch.ones(3), scene_midpoint=torch.zeros(3))
    hydrant_field2 = NGPField('wisp/configs/ngp_hydrant_new.yaml')
    car_field = NGPField('wisp/configs/ngp_car.yaml')
    car_field2 = NGPField('wisp/configs/ngp_car.yaml')
    
    params = np.load("../exp-params/right-2c-3h/seed-0.npy")
    params = torch.from_numpy(params).cuda()
    
    for field_ in [hydrant_field, hydrant_field1, hydrant_field2, car_field, car_field2]:
        p_ = make_functional(field_.pipeline.nef, param_filter=lambda name: "codebookcolor" in name)
        current_params = p_.param_vector
        print(current_params.shape, params.shape)
        set_weights(field_.pipeline.nef, p_, current_params + params[:current_params.shape[0]])
        params = params[current_params.shape[0]:]
    

    hydrant_loc = torch.tensor([-0.1, 0.00, 0.04, 0.]).cuda() 
    hydrant_loc2 = torch.tensor([-0.0899, 0.0, 0.0188, 0.]).cuda() 
    hydrant_loc3 = torch.tensor([-0.0899, -0.0766, 0.0188, 0.]).cuda() 
    
    car_loc = torch.tensor([-0.03, 0.12, 0.0188, np.pi/2]).cuda()
    car_loc2 = torch.tensor([0.008, -0.09, 0.0188, -np.pi/2]).cuda()
#                    
  #  field.add_obj_field(hydrant_field, hydrant_loc)
    field.add_obj_field(hydrant_field1, hydrant_loc2)
  #  field.add_obj_field(hydrant_field2, hydrant_loc3)
  #  field.add_obj_field(car_field, car_loc)
  #  field.add_obj_field(car_field2, car_loc2)

    print(hydrant_loc, car_loc)
   # field.set_transform_params(p)
    with torch.autograd.no_grad():
        camera = Camera(640, 360, 320)
        data1, depth = camera.read(field, location.cuda())
#        data2, depth = camera.read(field, location.cuda() + torch.tensor([-15., 10., -np.pi/2]).cuda())
        
#    data = torch.cat([data1[0], data2[0]], dim = 1)
    data = data1[0]
        
    return data.cpu().numpy()


