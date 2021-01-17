import torch

from cubeadv.policies.expert import Expert
from cubeadv.sim.dynamics import Dynamics
from cubeadv.sim.utils import PathMapCost, STARTS, STEPS, connect_carla
from cubeadv.fields import NGPField
from cubeadv.sim.sensors import Camera, CarlaCamera

from cubeadv.utils.plotting import colorline
import pytest

from test_utils import plotting_test, plot_road
import matplotlib.pyplot as plt
import matplotlib as mpl

starts = list(zip(STARTS, range(3)))
goals = list(zip(STARTS, range(3)))

def drive(start, pm, start_idx, goal_idx):
    expert = Expert(-2., pm)
    expert.load_params(1., 20., 50.)
    trajectory = []
    dynamics = Dynamics(0.01)

    x = torch.cat([start, pm.get_initial_lane_alignment()], dim=-1)
    x[:2] = pm.get_offset(x[:2], -4.)
    T = torch.arange(1000)
    for t in T:
        u = expert(x[:2].view(1, -1))
        x = dynamics(x, u).squeeze()
        trajectory.append(x)

    return torch.stack(trajectory)

@pytest.mark.parametrize("start,start_idx", starts)
@pytest.mark.parametrize("goal,goal_idx", goals)
@plotting_test
def test_expert_policy(start, goal, start_idx, goal_idx, request):
    carla = True
    if torch.allclose(start, goal):
        return None, None, None

    pm = PathMapCost.get_carla_town(start.view(1, -1), goal.view(1, -1))
    t = drive(start, pm, start_idx, goal_idx)
    fig, ax = plot_road(pm, t)

    return fig, ax, request.node.callspec.id

