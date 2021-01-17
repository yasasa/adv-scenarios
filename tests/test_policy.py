import torch
import os, sys
sys.path.append(os.environ["CARLA_PYTHON_PATH"])

from cubeadv.policies.expert import Expert
from cubeadv.sim.dynamics import Dynamics
from cubeadv.sim.utils import PathMapCost, STARTS, STEPS
from cubeadv.fields import NGPField, NGPComposeField
from cubeadv.sim.sensors import Camera
from cubeadv.sim.sensors.carla_camera import CarlaCamera, CarlaCameraCompose
from cubeadv.sim.utils import connect_carla
from cubeadv.fields.utils import normalize_ngp
from cubeadv.utils.plotting import colorline
from cubeadv.policies.cnn import Policy, PolicyDepth, PolicyNoStart
import pytest

from test_utils import plotting_test, driving_test, plot_road
import matplotlib.pyplot as plt
import matplotlib as mpl

import imageio
import numpy as np

starts = list(zip(STARTS, range(0, 3)))
goals = list(zip(STARTS, range(0, 3)))

@torch.no_grad()
def drive(start, pm, goal, start_idx, goal_idx, policy, render_fn, device):
    pm = pm.to(device)
    expert = Expert(2., pm)
    expert.load_params(1., 20., 50.)
    trajectory = []
    os = []
    dynamics = Dynamics(0.01)
    
    car_starts = torch.tensor([[105.4, 129.5],
                               [88.4, 110.5],
                               STARTS[2].numpy().tolist()])
    
    goal_one_hot = torch.nn.functional.one_hot(torch.tensor([goal_idx]), len(STARTS)).cuda()
    start_one_hot = torch.nn.functional.one_hot(torch.tensor([start_idx]), len(STARTS)).cuda()
    
#    goal = normalize_ngp(goal.cuda().view(1, -1), 1.)
   
    x = torch.cat([car_starts[start_idx].cuda(), pm.get_initial_lane_alignment()], dim=-1)
    x[:2] = pm.get_offset(x[:2], 0.)
    x = x.cuda()
    T = torch.arange(500)
    for t in T:
        o, depth = render_fn(x.view(1, -1))
        depth = depth.cuda()
        
        o = o.permute(0, 3, 1, 2).cuda()
        depth = depth.view(-1, 1, o.shape[-2], o.shape[-1])
        
        u = policy(o, goal_one_hot, depth, start_one_hot).view(-1)
        u_expert = expert(x[:2].view(1, -1)).cuda()
        x = dynamics(x, u).squeeze()
        trajectory.append(x.detach().cpu())
        os.append(o[0].permute(1, 2, 0).detach().cpu().numpy())
    
    return torch.stack(trajectory), os

@pytest.mark.parametrize("goal,goal_idx", goals)
@pytest.mark.parametrize("start,start_idx", starts)
@driving_test
def test_policy(start, goal, start_idx, goal_idx, request):
    if torch.allclose(start, goal):
        return None, None, None, None
    carla = True
    depth = False
    num_obj = 0
    
    pm = PathMapCost.get_carla_town(start.view(1, -1), goal.view(1, -1))
    
    if depth:
        policy = PolicyDepth()
        policy.load_state_dict(torch.load("adversarial-nerf-models/depth-policy.pt"))
    else:
        policy = PolicyNoStart()
        policy.load_state_dict(torch.load('/media/ssd/users/yasasa/carla-data-rand-obj-9/policy_epoch_299_no_start.pt'))
      #  policy.load_state_dict(torch.load('../experiments/fixed-policy/policy_epoch_159_no_start.pt'))
        
    policy = policy.cuda()
    
    if carla:
        _, world = connect_carla("localhost", 2000)
        field = CarlaCameraCompose(world, 200, 66, 100)
        render_fn = lambda x: field.read(x)
    else:
        
        field = NGPComposeField(NGPField("wisp/configs/ngp_nerf_bg_new.yaml"))
        field.scene_field.pipeline.nef.ignore_view_dir = False
                     
        camera = Camera(200, 66, 100)
        render_fn = lambda x: camera.read(field, x)
    
    hydrant_field = NGPField('wisp/configs/ngp_hydrant_new.yaml', scene_scale=torch.ones(3), scene_midpoint=torch.zeros(3))
    car_field = NGPField('wisp/configs/ngp_car.yaml', scene_scale=torch.ones(3), scene_midpoint=torch.zeros(3))
    
    min12 = torch.tensor([-0.0799, -0.0866, 0.0188, 0.,  -0.0799, -0.0866, 0.0188, 0.,  -0.0799, -0.0866, 0.0188, 0., 0.0799, 0.045, 0.0188, 0.,  0.008, -0.09, 0.0188, -1.57079632])
    max12 = torch.tensor([-0.0799, 0.12, 0.0188, 0.,  -0.0799, 0.12, 0.0188, 0.,  -0.0799, 0.12, 0.0188, 0., 0.12, 0.045, 0.0188, 0.,  0.008, 0.0, 0.0188, -1.57079632])
    min01 = torch.tensor([-0.0799, -0.0866, 0.0188, 0., -0.0799, -0.0866, 0.0188, 0., -0.0799, -0.0866, 0.0188, 0., -0.03, -0.0866, 0.0188, 1.57079632, 0.008, -0.09, 0.0188, -1.57079632])
    max01 = torch.tensor([-0.0799, 0.12, 0.0188, 0., -0.0799, 0.12, 0.0188, 0., -0.0799, 0.12, 0.0188, 0., -0.03, 0.12, 0.0188, 1.57079632, 0.008, -0.01, 0.0188, -1.57079632])
    min10 = torch.tensor([-0.0799, -0.0866, 0.0188, 0.,  -0.0799, -0.0866, 0.0188, 0.,  -0.0799, -0.0866, 0.0188, 0., 0.0799, 0.045, 0.0188, 0.,  0.008, -0.12, 0.0188, -1.57079632])
    max10 = torch.tensor([-0.0799, 0.12, 0.0188, 0.,  -0.0799, 0.12, 0.0188, 0.,  -0.0799, 0.12, 0.0188, 0., 0.12, 0.045, 0.0188, 0.,  0.008, -0.09, 0.0188, -1.57079632])
    mins = [min10, min01, min12]
    maxs = [max10, max01, max12]
    
    minv = mins[goal_idx]
    maxv = maxs[goal_idx]
    
    v = torch.rand_like(minv) * (maxv - minv) + minv
    v = torch.tensor([-0.0799, 0.0866, 0.0188, 0.,  -0.0799, 0.0, 0.0188, 0.,  -0.0799, -0.0766, 0.0188, 0., 0.0799, 0.045, 0.0188, 0.,  0.008, -0.09, 0.0188, -1.57079632]
)


    hydrant_loc, hydrant_loc1, hydrant_loc2, car_loc, car_loc2 = v.cuda().split(4)

    field.add_obj_field(hydrant_field, hydrant_loc)
    field.add_obj_field(hydrant_field, hydrant_loc1)
    field.add_obj_field(hydrant_field, hydrant_loc2)
    field.add_obj_field(car_field, car_loc)
    field.add_obj_field(car_field, car_loc2)
    device ='cuda'
    
    t, o = drive(start, pm, goal, start_idx, goal_idx, policy, render_fn, device)
    fig, ax = plot_road(pm, t)
    
    suffix = "carla" if carla else "nerf"
    return fig, ax, o, f"{request.node.callspec.id}_{suffix}"
    
    