import torch
from torch.utils import checkpoint as ck
import scipy
import numpy as np
from torchviz import make_dot
import sys

from .optimizer import CubeOptimizer

import time

class ShootingDescent(CubeOptimizer):
    def __init__(self,
            dynamics,
            sensor,
            policy,
            step_cost,
            save_path,
            state_dim=3,
            action_dim=1,
            cube_dim=4,
            integration_timesteps=200,
            descent_iterations=100):
        super(ShootingDescent,
                self).__init__(dynamics, sensor, policy, step_cost, save_path,
                        state_dim, action_dim, cube_dim)
        self._T = integration_timesteps
        self._itr = descent_iterations

    def _step(self, c, x0):
        x = x0

        total_cost = torch.tensor(0.)

        for i in range(self._T):
            o = self._sensor(x, c)
            u = self._policy.pi( o, tensor=True)
            xn = self._dynamics(x, u)

            step_cost = self._step_cost.cost(xn, u)

            total_cost += step_cost
            del o, u, step_cost
            x = xn

        return -total_cost

    def cost(self, c, x0):
        cost = self._step(c, x0)
        return cost

    def save_graph(self, l, params=None):
        t = make_dot(l, params=params)
        t.render('output.gv', view=False)


    def run(self, c0, x0):
        x0 = torch.from_numpy(x0).to(dtype=torch.float)
        eps = 1e-4

        f = lambda c: self.cost(c, x0)

        def objective(x):
            c = f(x)
            print(c, x)
            return c

        c_ = torch.from_numpy(c0).to(dtype=torch.float).requires_grad_(True)

        optimizer = torch.optim.Adam([c_], lr = 0.0001)
        for itr in range(self._itr):
            optimizer.zero_grad()
            loss = f(c_)
            loss.backward()
#            self.save_graph(loss)
            optimizer.step()
            # TODO: use some better way to print the vector based on cube dim
            li = loss.item()
            print("[Iteration {:d}:] [{:.2f}, {:.2f}] Cost: {:.4f}".format(itr+1, c_[0], c_[1], loss.item()))
            del loss
            torch.cuda.empty_cache()

