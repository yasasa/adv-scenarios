from curses import A_ALTCHARSET
import torch
import numpy as np

import scipy.special as sp
import scipy.interpolate as interp


STARTS = torch.tensor([
          [129.4, 132.5],
          [90.4, 102.],
          [90.4, 155.]])

STEPS = torch.tensor([[0, 1000, 1000], [1000, 0, 1000], [1000, 1000, 0]], dtype=torch.int32)


def cross2d(x, y):
    return x[..., 0]*y[..., 1] - x[..., 1]*y[..., 0]

def reflect(rp, rv, x):
    v = x - rp
    drv = (rv * v).sum(dim=-1)

    rrv = 2 * drv[..., None] * rv - v
    return rp + rrv


class PathMapCost:
    """ Provides cost of distance from closest center lane for the given lane map"""
    def __init__(self, pts):
        self.batch_size = pts.shape[0]
        self.add_segments(pts)

    def add_segments(self, pt : torch.Tensor):
        diff = pt[:, 1:] - pt[:, :-1]  # [batch, segments-1, 2]
        self.lanes = diff / diff.norm(dim=-1, keepdim=True)
        self.points = pt

    def __getitem__(self, idx):
        return PathMapCost(self.lanes[idx])

    def to(self, *args, **kwargs):
        points = self.points.to(*args, **kwargs)
        if points is not self.points:
            return PathMapCost(points)
        return self
    
    def offset_map(self, offset):
        lv = self.lanes.clone()
        lv[:, :, [0, 1]] = lv[:, :, [1, 0]]
        lv[:, :, 0] *= -1
        points = self.points.clone()
        points[:, 1:] += offset * lv
        points[:, 0] += offset * lv[:, 0]
        return points

    def projection(self, point):
        v = point[:,  None, :] - self.points[:, :-1] # [batch, segments, 2]
        max_len = (self.points[:, 1:] - self.points[:, :-1]).norm(dim=-1)

        londist = (v * self.lanes).sum(dim=-1)

        londist = londist.clamp(min=torch.zeros_like(max_len), max=max_len) # projection to lanes [batch, segments]
        sign = torch.sign(cross2d(v, self.lanes)) #[batch, segment]

        p = self.points[:, :-1] + londist[..., None] * self.lanes # projected points [batch, segments, 2]

        latv = (point[:, None, :] - p).norm(dim=-1)
        sign[torch.isclose(sign, torch.zeros_like(sign))] = 1.
        latdist = sign * latv

        segment = latv.argmin(dim=1) # [batch,]
        latdist = latdist[torch.arange(point.shape[0]), segment]
        londist = londist[torch.arange(point.shape[0]), segment]

        return londist, latdist, segment

    def get_lateral_vector(self):
        lv = self.lanes[:, 0].clone()
        lv[:, [0, 1]] = lv[:, [1, 0]]
        lv[:, 0] *= -1
        return lv

    def get_offset(self, x, offset):
        # Assume z pointing down to get good offsets in the carla coordinates
        return x - self.get_lateral_vector() * offset

    def old_projection(self, point):
        a, b, c = self.projection(point)
        return b

    def signed_dist(self, x):
        dist, lat_dist, _ = self.projection(x)
        return lat_dist

    def cost(self, x, u):
        if x.dim() == 1:
            x = x.unsqueeze(0) # add batch dim
        return self.projection(x)[1].abs()

    def Jx(self, x, u):
        x_ = torch.tensor(x, requires_grad=True).float()
        y = self.project_to_path(x_[:2])[0]
        jac = torch.autograd.grad(y, x_)[0]
        return jac.view(1, -1).detach().numpy()

    def Ju(self, x, u):
        return np.zeros_like(u)

    def get_initial_lane_alignment(self):
        theta = torch.arctan2(self.lanes[:, 0, 1], self.lanes[:, 0, 0])
        return theta

    def __call__(self, x, u):
        return self.cost(x, u)

    @staticmethod
    def get_carla_town(start, goal):
        center = torch.tensor([90.4, 132.5])
        center_dir = torch.tensor([-1., 0])

        curve = torch.tensor([[129.4, 132.5], [96.4, 132.5], [93.4, 130.7], [91.2, 128.5], [90.4, 125.5], [90.4, 102.]])

        straight = torch.stack([STARTS[2], STARTS[1]])
        straight = torch.nn.functional.interpolate(straight.T.unsqueeze(0), curve.shape[0], mode='linear', align_corners=True).squeeze().T

        pm1 = curve
        pm2 = reflect(center,  center_dir, curve)

        pm3 = pm1.flip(0)
        pm4 = straight.flip(0)

        pm5 = pm2.flip(0)
        pm6 = straight

        pm = torch.stack([pm1, pm1, pm2, pm3, pm3, pm4, pm5, pm6, pm6])

        # Going to have redundancies i.e start == goal, if the argmin below doesn't take them out then by default it will just use the tmp element
        _s1 = STARTS.view(3, 1, 2).repeat(1, 3, 1)
        _s2 = STARTS.view(1, 3, 2).repeat(3, 1, 1)
        start_goals_possible = torch.cat([_s1.view(-1, 2), _s2.view(-1, 2)], dim=-1)

        start_goals = torch.cat([start, goal], dim=-1)
        dists = start_goals[:, None, :] - start_goals_possible[None]
        dists = dists.norm(dim=-1)
        min_dist = dists.argmin(1)
        maps = pm[min_dist]
        return PathMapCost(maps)
    
    @staticmethod
    def get_square_map(ur, ll):
        points = [ll, [ll[0], ur[1]], ur, [ur[0], ll[1]], ll]
        points = torch.tensor(points)
        return PathMapCost(points.view(1, 5, 2))