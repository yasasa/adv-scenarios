import torch
import os, sys

from cubeadv.policies.expert import Expert
from cubeadv.sim.dynamics import Ackermann
from cubeadv.sim.utils import PathMapCost, STARTS
from cubeadv.fields import NGPField, NGPComposeField
from cubeadv.sim.sensors import Camera
from cubeadv.sim.utils import connect_carla
from cubeadv.policies.cnn import Policy, get_img_transform
import cubeadv.utils as util
import pytest

import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import imageio

from test_utils import driving_test

starts = list(zip(STARTS, range(3)))
goals = list(zip(STARTS, range(3)))

@torch.no_grad()
def drive(policy, render_fn, device):
    trajectory = []
    os = []
    dynamics = Ackermann(0.01)
    
    x = torch.tensor([2.5/32., 0., np.pi/2])
    x = x.cuda()
    T = torch.arange(1000)
    for t in T:
        o, _ = render_fn(x.view(1, -1), t)
        u = policy(o).view(-1)
        print(t, u)
        x = dynamics(x, u).squeeze()
        trajectory.append(x.detach().cpu())
        os.append(o[0].permute(1, 2, 0).detach().cpu().numpy())
    
    return torch.stack(trajectory), os

@driving_test
def test_real_policy(request):
    policy = Policy()
    policy.load_state_dict(torch.load('adversarial-nerf-models/my580-policy.pt'))
    policy = policy.cuda()
    
    #transform = util.get_permutation(torch.tensor([0, 2, 1]), homog=True)
    #transform[:, [1, 2]] *= -1
    field = NGPComposeField(NGPField("wisp/configs/nerf-my-580-2.yaml", scene_scale=8*torch.ones(3), scene_midpoint=torch.zeros(3), transform=transform))
    field.scene_field.pipeline.nef.ignore_view_dir = False
                 
    camera = Camera(336, 188, [172, 157])
    tr = get_img_transform()
    
    
    def render_fn(x, count):
        o, d = camera.read(field, x)
        imageio.imwrite(f"debug-images/im_{count}.png", (254*o.squeeze().cpu()).byte())
        o = tr(o.squeeze()).cuda()[None]
        return o, d
        
    device ='cuda'
    
    t, o = drive(policy, render_fn, device)
    fig, ax = plt.subplots()
    ax.plot(t[:, 0].detach().cpu(), t[:, 1].detach().cpu())
    
    return fig, ax, o, f"test_policy"
    
    