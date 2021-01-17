import torch

from cubeadv.policies.expert import Expert
from cubeadv.sim.dynamics import Dynamics
from cubeadv.sim.utils import PathMapCost, STARTS, STEPS
from cubeadv.fields import NGPField
from cubeadv.fields.utils import normalize_ngp
from cubeadv.sim.sensors import Camera, CarlaCamera
from cubeadv.sim.utils import connect_carla

from cubeadv.utils.plotting import colorline
from cubeadv.policies.cnn import Policy, PolicyNoStart
import pytest

from test_utils import plotting_test, driving_test, plot_road
import matplotlib.pyplot as plt
import matplotlib as mpl

import imageio

starts = [(torch.tensor([88.4, 114.5]), 1)]
goals = [(STARTS[0], 0)]

@torch.no_grad()
def drive(start, pm, goal, start_idx, goal_idx, policy, render_fn, device):
    pm = pm.to(device)
    expert = Expert(2., pm)
    expert.load_params(1., 20., 50.)
    dynamics = Dynamics(0.01)

    goal_one_hot = torch.nn.functional.one_hot(torch.tensor([goal_idx]), len(STARTS)).cuda()
    start_one_hot = torch.nn.functional.one_hot(torch.tensor([start_idx]), len(STARTS)).cuda()

    goal = normalize_ngp(goal.cuda().view(1, -1), 1.)

    x = torch.cat([start.cuda(), pm.get_initial_lane_alignment()], dim=-1)
    x[:2] = pm.get_offset(x[:2], 0.)
    x = x.cuda()

    x_expert = x.clone()

    T = torch.arange(1000)
    cost = 0
    for t in T:
        o, _ = render_fn(x.view(1, -1))
        o = o.permute(0, 3, 1, 2).cuda()

        u = policy(o, goal_one_hot, start_one_hot).view(-1)
        u_expert = expert(x_expert[:2].view(1, -1)).cuda()

        x = dynamics(x, u).squeeze()
        x_expert = dynamics(x_expert, u_expert).squeeze()

        c = pm.cost(x[:2], None)
        c_expert = pm.cost(x_expert[:2], None)
        
        cost += (c - c_expert)**2

    return cost

@pytest.mark.parametrize("goal,goal_idx", goals)
@pytest.mark.parametrize("start,start_idx", starts)
@plotting_test
def test_policy_robustness(start, goal, start_idx, goal_idx, request):
    if torch.allclose(start, goal):
        return None, None, None, None
    carla = False
    pm = PathMapCost.get_carla_town(start.view(1, -1), goal.view(1, -1))
    policy = PolicyNoStart()
    policy.load_state_dict(torch.load('adversarial-nerf-models/rgb-policy.pt'))
    policy = policy.cuda()
    policy = policy.train()

    if carla:
        _, world = connect_carla("localhost", 2000)
        camera = CarlaCamera(world, 200, 66, 100)
        render_fn = lambda x: camera.read(x, convert=True)
    else:
        field = NGPField("wisp/configs/ngp_nerf_bg_new.yaml", True)
        camera = Camera(200, 66, 100)
        render_fn = lambda x: camera.read(field, x)

    device ='cuda'

    noise_std = torch.arange(50) * 1e-2
    data_mu = []
    data_std = []
    
    for std in noise_std:
        def render_fn_(x):
            y, _ = render_fn(x)
            return y + std * torch.randn_like(y), None
        
        cost = []
        for i in range(5):
            cost_ = drive(start, pm, goal, start_idx, goal_idx, policy, render_fn_, device)
            cost.append(cost_.squeeze().cpu())
        cost = torch.stack(cost)
        data_mu.append(cost.mean())
        data_std.append(cost.std())
    
    data_mu = torch.stack(data_mu)
    data_std = torch.stack(data_std)
    
    fig, ax = plt.subplots()
    ax.plot(noise_std, data_mu)
    ax.fill_between(noise_std, data_mu - data_std, data_mu + data_std, alpha=0.5)
    

    suffix = "carla" if carla else "nerf"
    return fig, ax, f"{request.node.callspec.id}_{suffix}"

