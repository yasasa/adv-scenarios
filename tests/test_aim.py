import torch
import os, sys
sys.path.append(os.environ["CARLA_PYTHON_PATH"])
sys.path.append(os.environ["CARLA_GARAGE_PATH"])

from cubeadv.policies.expert import Expert
from cubeadv.sim.dynamics import TeleportDynamics, SPEED
from cubeadv.sim.utils import PathMapCost, STARTS, STEPS
from cubeadv.fields import NGPField, NGPComposeField
from cubeadv.sim.sensors import Camera
from cubeadv.sim.sensors.carla_camera import CarlaCamera
from cubeadv.sim.utils import connect_carla
from cubeadv.fields.utils import normalize_ngp
from cubeadv.utils.plotting import colorline
from cubeadv.policies.cnn import Policy, PolicyDepth, PolicyNoStart
import pytest

from sensor_agent_simple import SensorAgentSimple

from test_utils import plotting_test, driving_test, plot_road
import matplotlib.pyplot as plt
import matplotlib as mpl

import imageio
import numpy as np

starts = list(zip(STARTS, range(3)))
goals = list(zip(STARTS, range(3)))

@torch.no_grad()
def drive(start, pm, goal, start_idx, goal_idx, policy, render_fn, device):
    pm = pm.to(device)
    expert = Expert(2., pm)
    expert.load_params(1., 20., 50.)
    trajectory = []
    os = []
    dynamics = TeleportDynamics(0.1)
    
    x = torch.cat([start.cuda(), pm.get_initial_lane_alignment()], dim=-1)
    x[:2] = pm.get_offset(x[:2], -1.)
    x = x.cuda()
    T = torch.arange(500)
    for t in T:
        o, depth = render_fn(x.view(1, -1))
        depth = depth.cuda()
        
      #  o = o.permute(0, 3, 1, 2).cuda()
      #  depth = depth.view(-1, 1  o.shape[-2], o.shape[-1])
      
        policy_input = {
            "rgb_front" : o.squeeze().cuda()* 255.,
            "speed" : SPEED,
            "target": STARTS[goal_idx].cuda(),
            "imu" : torch.rad2deg(x[-1]),
            "gps" : x[:2],
        }
        u = policy.run_step(policy_input, 0)
      #  print(u)
        u_expert = expert(x[:2].view(1, -1)).cuda()
        x = dynamics(x, u[0]).squeeze()
        trajectory.append(x.detach().cpu())
        os.append(o[0].detach().cpu().numpy())
    return torch.stack(trajectory), os

@pytest.mark.parametrize("goal,goal_idx", goals)
@pytest.mark.parametrize("start,start_idx", starts)
@driving_test
def test_aim(start, goal, start_idx, goal_idx, request):
    carla = True
    if torch.allclose(start, goal):
        return None, None, None, None
    
    pm = PathMapCost.get_carla_town(start.view(1, -1), goal.view(1, -1))
 
    config = "/media/ssd/users/yasasa/pretrained_models/lav/aim_02_05_withheld_1"
    policy = SensorAgentSimple(config)
        
    
    _, world = connect_carla("localhost", 2000)
    camera = CarlaCamera(world, 1024, 256, 358.5)
    render_fn = lambda x: camera.read(x, convert=True)
    
    device ='cuda'
    
    t, o = drive(start, pm, goal, start_idx, goal_idx, policy, render_fn, device)
    fig, ax = plot_road(pm, t)
    
    suffix = "carla" if carla else "nerf"
    return fig, ax, o, f"{request.node.callspec.id}_{suffix}"
    
    