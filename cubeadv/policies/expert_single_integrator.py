import torch
import numpy as np


class ExpertSingleIntegrator:
    def __init__(self, target_dist, lane_map, max_windup=1.):
        self._lane_map = lane_map

        self._Kp = 1.
        self._Ki = 0.
        self._Kd = 0.

        self._acc = 0.
        self._prev = None

        self.target_dist = target_dist
        self.max_windup = max_windup

    def load_params(self, Kp, Ki, Kd):
        self._Kp = Kp
        self._Ki = Ki
        self._Kd = Kd

    def reset(self):
        self._prev = None
        self._acc = 0.

    def _run(self, x):
        sd = self._lane_map.signed_dist(x, self.target_dist)
        e = sd
        ed = 0
        if self._prev is not None:
            ed = e - self._prev

        self._acc = max(min(self._acc, self.max_windup), -self.max_windup)

        u = self._Kp * e + self._Kd*ed + self._Ki*self._acc
        self._prev = e

        u = max(-1.0, min(u, 1.0)) + x[2]
#         print(x, e, u)
#         return np.array([u])
        return np.array([np.cos(u), np.sin(u)])


    def __call__(self, x):
        return self._run(x)

