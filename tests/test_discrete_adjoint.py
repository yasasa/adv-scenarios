
import torch

from cubeadv.policies.expert import Expert
from cubeadv.sim.dynamics import Dynamics
from cubeadv.sim.utils import PathMapCost, STARTS, STEPS
from cubeadv.fields import NGPField, NGPComposeField
from cubeadv.fields.utils import normalize_ngp
from cubeadv.sim.sensors import Camera, CarlaCamera
from cubeadv.sim.utils import connect_carla

from cubeadv.opt import discrete_adjoint

from cubeadv.utils.plotting import colorline
from cubeadv.policies.cnn import Policy, PolicyNoStart
import pytest

from test_utils import plotting_test, driving_test, plot_road
import matplotlib.pyplot as plt
import matplotlib as mpl

import imageio

starts = list(zip(STARTS, range(3)))
goals = list(zip(STARTS, range(3)))

def dynamics(x, p):
    xdotdot = -x[:, 1] * p[:, 0]
    xdot = x[:, 1]
    return torch.stack([xdot, xdotdot], dim=-1)
    
def F(i, t2, t1, x2, x1, p):
    xdot = dynamics(x1, p)
    return x1 + (t2 - t1)*xdot
    
def cost(x):
    return (x*x).sum(dim=-1)
    
@pytest.mark.parametrize("batch", [1, 4, 8])
def test_policy(batch):
    x0 = torch.randn(batch, 2).double()
    p0 = torch.randn(batch, 1).abs().requires_grad_(True).double()
    print(p0)
    
    T = torch.linspace(0, 1, 100).double()
    f = lambda p: discrete_adjoint(F, cost, x0, T, p).sum(dim=0)
    torch.autograd.gradcheck(f, p0)
    
    
    