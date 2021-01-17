import os, sys
sys.path.append(os.environ["CARLA_PYTHON_PATH"])
from pathlib import Path
import time
import signal
from tkinter.tix import ButtonBox

import pytest
import subprocess

import carla
import imageio
import torch
from cubeadv.fields.ngp import NGPField

from cubeadv.utils import make_functional, set_weights
import cubeadv.utils as util

from cubeadv.sim.sensors import carla_camera
from test_utils import *
import numpy as np


@pytest.fixture
def carla_world():
    base_path = os.environ.get("CARLA_PATH")
    sim_path = Path(base_path, "CarlaUE4.sh")

    assert(sim_path.exists())
    proc = subprocess.Popen(" /Game/Carla/Maps/Town01", executable=sim_path, preexec_fn=os.setsid)

    try:
        for attempt in range(10):
            time.sleep(2)
            try:
                client = carla.Client("localhost", 2000)
                world = client.load_world('Town01')
                world = client.reload_world()
                break
            except Exception as e:
                pass

        else:
            assert(False and "Failed to connect to carla")

        yield world
    finally:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)

@pytest.mark.parametrize("location", torch.tensor([[100.4, 132.5, np.pi - 0.1]]))
@image_comparison_test
@torch.no_grad()
def test_carla_camera(carla_world, location):
    box = True   
    hydrant_field = NGPField('wisp/configs/ngp_hydrant_new.yaml')
    car_field = NGPField('wisp/configs/ngp_car.yaml')

    hydrant_field = NGPField('wisp/configs/ngp_hydrant_new.yaml', scene_scale=torch.ones(3), scene_midpoint=torch.zeros(3))
    hydrant_field1 = NGPField('wisp/configs/ngp_hydrant_new.yaml')
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
        
    hydrant_loc = torch.tensor([-0.0799, 0.0866, 0.0188, 0.]).cuda() 
    hydrant_loc2 = torch.tensor([-0.0899, 0.0, 0.0188, 0.]).cuda() 
    hydrant_loc3 = torch.tensor([-0.0899, -0.0766, 0.0188, 0.]).cuda() 
    
    car_loc = torch.tensor([-0.03, 0.12, 0.0188, np.pi/2]).cuda()
    car_loc2 = torch.tensor([0.008, -0.09, 0.0188, -np.pi/2]).cuda()
                     
    camera = carla_camera.CarlaCameraCompose(carla_world, 1280, 720, 640, box=box)
   #
    camera.add_obj_field(hydrant_field, hydrant_loc)
    camera.add_obj_field(hydrant_field, hydrant_loc2)
    camera.add_obj_field(hydrant_field, hydrant_loc3)
   # camera.add_obj_field(car_field, car_loc)
   #camera.add_obj_field(car_field2, car_loc2)
    
    location = torch.tensor(location).cuda()
    image1, d1 = camera.read(location)
    image2, d2 = camera.read(location + torch.tensor([15., 15., np.pi/2]).cuda())
#    image1 = d1
#    image2 = d2
    
    
    data = image1[0].cpu() #torch.cat([image1[0].cpu(), image2[0].cpu()], dim = 1)

    return data


