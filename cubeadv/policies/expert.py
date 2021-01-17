import torch
import numpy as np

class Expert:
    def __init__(self, target_dist, lane_map, max_windup=1.):
        self._lane_map = lane_map

        self._Kp = 1.
        self._Ki = 0.
        self._Kd = 0.

        self._acc = torch.tensor(0.)
        self._prev = None

        self.target_dist = target_dist
        self.max_windup = max_windup

    def load_params(self, Kp, Ki, Kd):
        self._Kp = Kp
        self._Ki = Ki
        self._Kd = Kd

    def reset(self):
        self._prev = None
        self._acc = torch.tensor(0.)

    def _run(self, x):
        sd = self._lane_map.signed_dist(x) - self.target_dist

        e = sd
        ed = 0
        if self._prev is not None:
            ed = e - self._prev
        self._acc = self._acc.clamp(min=-self.max_windup, max=self.max_windup)

        u = self._Kp * e + self._Kd*ed + self._Ki*self._acc
        self._prev = e

        u = u.clamp(min=-1, max=1)
        return u

    def __call__(self, x):
        return self._run(x)

