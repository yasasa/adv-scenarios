import os
import sys
import numpy as np

import torch

MAX_STEERING = 35
SPEED = 5

class SingleDynamics:
    def __init__(self, dt):
        self.dt = dt

    def __call__(self, s, u):
        # u = [v_x, v_y, 0]
        return s + SPEED * self.f(s, u) * self.dt

    def f(self, s, u):
        return torch.stack([u[0], u[1], torch.tensor(0.0).type_as(u)])