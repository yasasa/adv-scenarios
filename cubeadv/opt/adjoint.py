import torch
from torch.utils import checkpoint as ck
import scipy
import numpy as np
from torchdiffeq import odeint_adjoint as odeint
from torchviz import make_dot
import sys

from .optimizer import CubeOptimizer

import time

class RunnerModule(torch.nn.Module):
    def __init__(self, cube_params, sensor, policy, dynamics):
        super().__init__()
        self.c = torch.nn.Parameter(cube_params)
        self._sensor = sensor
        self._policy = policy
        self._dynamics = dynamics


    def forward(self, t, x, **args):
        with torch.profiler.record_function("sensor"):
            o = self._sensor(x, self.c)
           # print(o.shape)
        with torch.profiler.record_function("policy"):
            u = self._policy.pi(o, tensor=True)
        with torch.profiler.record_function("dynamics"):
            xdot = self._dynamics.f(x, u)
        return xdot

class AdjointDescent(CubeOptimizer):
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
            descent_iterations=100,
            end_time=1.):
        super(AdjointDescent,
                self).__init__(dynamics, sensor, policy, step_cost, save_path,
                        state_dim, action_dim, cube_dim)
        self._T = integration_timesteps
        self._itr = descent_iterations
        self._tn = end_time


        self.T = torch.linspace(0, self._tn, self._T)

    def C(self, X):
        x0 = X[0]
        xs = X[1:] # ignore starting state

        l = [self._step_cost.cost(x, None) for x in xs]
        return -torch.sum(torch.stack(l))

    def save_graph(self, l, params=None):
        t = make_dot(l, params=params)
        t.render('output.gv', view=False)

    def integrate(self, f, x0, T, method="euler"):
        X = odeint(f, x0, T, method="euler")
        loss = self.C(X)
        return loss

    @torch.no_grad()
    def finite_diff(self, l0, c0, x0, eps):
        n = c0.shape[0]
        In = torch.eye(n).type_as(c0)
        grad = torch.zeros_like(c0)
        for i in range(n):
            f = RunnerModule(c0 + eps * In[i], self._sensor, self._policy, self._dynamics)
            grad[i] = (self.integrate(f, x0, self.T) - l0) / eps

        return grad


    def run(self, c0, x0, bounds_min=None, bounds_max=None, itr=200, lr=1e-1, noise_std_dev=0, fd=False, line_search=None, forward_chunk_size=16384, backward_chunk_size=256):
        c0 = torch.tensor(c0, dtype=torch.float32).cuda()
        x0 = torch.tensor(x0, dtype=torch.float32).cuda()
        if bounds_min is not None:
            bounds_min = torch.tensor(bounds_min, dtype=torch.float32).cuda()
            bounds_max = torch.tensor(bounds_max, dtype=torch.float32).cuda()

        f = RunnerModule(c0, self._sensor, self._policy, self._dynamics)
        f.c.register_hook(lambda x: x + noise_std_dev * torch.randn_like(x))
        if line_search:
            optimizer = torch.optim.LBFGS(f.parameters(), lr=lr, line_search_fn=line_search)
        else:
            optimizer = torch.optim.Adam(f.parameters(), lr = lr)

        for itr in range(itr):
            li = []
            def closure():
                optimizer.zero_grad()

                t0 = time.time()
                self._sensor._world.args.chunk_size = 16384
                loss = self.integrate(f, x0, self.T)
                t1 = time.time()
                print("Forward pass took: %.4f" % (t1-t0))

                t2 = time.time()
                if fd:
                    grad = self.finite_diff(loss, f.c.clone().detach(), x0, lr)
                    f.c.grad = grad
                else:
                    self._sensor._world.args.chunk_size = 256 # Change chunk size to not run out of memory
                    loss.backward()
                t3 = time.time()
                print("Backward pass took: %.4f" % (t3-t2))
                li.append(loss.item())
                return loss
            optimizer.step(closure)

            if bounds_max is not None:
                f.c.clamp(max=bounds_max) # Project parameters to bounds
            if bounds_min is not None:
                f.c.clamp(min=bounds_min)
            li = li[-1]

            c_ = f.c
            with np.printoptions(precision=3, suppress=True):
                print("[Iteration {:d}:] Cost: {:.4f} Cubes: [{}]".format(itr+1, li, c_.detach().cpu().numpy()))
            yield li, c_

