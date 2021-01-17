import torch

SPEED = 0.5 * 32
BASELINE = 0.5 * 32

class Ackermann:
    def __init__(self, dt):
        self.dt = dt

    def __call__(self, s, u):
        return s + self.f(s, u) * self.dt

    def f(self, s, u, v=SPEED, l=BASELINE):
        if s.dim() == 1:
            s = s.unsqueeze(0)
        return torch.stack([v * torch.cos(-s[:, 2]), v * torch.sin(-s[:, 2]), v * torch.tan(u) / l], dim=-1)

        
