from multiprocessing.sharedctypes import Value
import os
import sys
import numpy as np

import torch

MAX_STEERING = 35
SPEED = 5

class Dynamics:
    def __init__(self, dt):
        self.dt = dt

    def __call__(self, s, u):
        return s + self.f(s, u) * self.dt

    def f(self, s, u):
        if s.dim() == 1:
            s = s.unsqueeze(0)
        return torch.stack([SPEED * torch.cos(s[:, 2]), SPEED * torch.sin(s[:, 2]), u], dim=-1)

class TeleportDynamics:
    def __init__(self, dt):
        self.dt = dt

    def __call__(self, s, u):
        return self.f(s, u)

    def f(self, s, u):
        if s.dim() == 1:
            s = s.unsqueeze(0)
        v = u[0, :2]
        
        ys = (-s[..., 2]).sin();
        yc = (-s[..., 2]).cos();
        vx = v.clone()
        vx[[0, 1]] = vx[[1, 0]]
        vx[1] = vx[1]*-1
        

        vr = yc*v + ys*vx
        
        angle = ((torch.atan2(vr[..., 1], vr[..., 0]).unsqueeze(-1) - s[..., 2]) + np.pi) % (2*np.pi) - np.pi 
        newx = s[..., :2] + vr
        
        
        
        offset = torch.cat([vr.unsqueeze(0), angle.unsqueeze(-1)], dim=-1)
        sn = s + offset*self.dt
        
        print(v, vr, sn, s, angle)
        return sn
        
        return torch.stack([SPEED * torch.cos(s[:, 2]), SPEED * torch.sin(s[:, 2]), u], dim=-1)