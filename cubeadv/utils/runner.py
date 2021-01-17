import os

import torch
import numpy as np
from torchdiffeq import odeint_adjoint, odeint
import imageio
from cubeadv.nerf.src.utils import to8b

class Runner:
    def __init__(self, sensor, dynamics, policy, cost_fn, expert=None):
        self.sensor = sensor
        self.dynamics = dynamics
        self.policy = policy
        self.expert = expert
        self.cost = cost_fn

    @torch.no_grad()
    def run_expert(self, x0, c, T):
        if self.expert is None:
            return

        cost = 0
        x = torch.from_numpy(x0).to(self.device)
        save_x = []

        for i in range(T):
            up = self.expert(x.cpu().numpy())
            u = torch.from_numpy(up).type_as(x).squeeze()
            save_x.append(x)
            x = self.dynamics(x, u).float()

        return save_x

    def cost(self, X):
        x0 = X[0]
        xs = X[1:] # ignore starting state

        l = [self.cost(x, None) for x in xs]
        return -torch.sum(torch.stack(l))

    def run_steps(self, x0, c, T, dt, cff=False):
        xs = torch.zeros(T+1, x0.shape[-1]).type_as(x0)
        xs[0] = x0
        cost = torch.tensor(0.).type_as(x0)

        for i in range(1, T+1):
            xs[i] = xs[i-1] + dt*self.step(0, xs[i-1], c)
            cost += self.cost(xs[i], None)
            print(i, xs[i])

        return cost, xs

    @torch.no_grad()
    def step(self, t, x, c):
        o = self.sensor(x, c)
        u = self.policy(o)
        return self.dynamics.f(x, u)


    def run_odeint(self, x0, c, tn, T, method="euler"):
        T_ = torch.linspace(0, tn, T).type_as(x0)
        X = odeint(lambda t, x: self.step(t, x, c), x0, T_, method=method)
        cost = self.cost(X)
        return cost, X

    def run_adjoint(self, x0, c, tn, T, method="euler"):
        T_ = torch.linspace(0, tn, T).type_as(x0)
        X = odeint_adjoint(lambda t, x: self.step(t, x, c), x0, T_,
                           method=method, adjoint_params=(c))
        cost = self.cost(X)
        return cost, X

    @torch.no_grad()
    def camera_runner(self, camera, xs, c, path):
        for i, x in enumerate(xs):
            rgb = camera.read(x, c)
            rgb = rgb.detach().cpu().numpy()

            brgb = to8b(rgb[:, :, :3].transpose(1, 0, 2))

            imageio.imwrite(os.path.join(path, "rgb_{}.png".format(i)), brgb)
