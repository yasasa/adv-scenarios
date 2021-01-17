import torch

from cubeadv.fields import NGPField, NGPComposeField
from cubeadv.fields.base_rf import MockRF
from cubeadv.sim.sensors import Camera, Lidar

from cubeadv.opt import discrete_adjoint
from cubeadv.sim.dynamics import Dynamics
from cubeadv.sim.utils import PathMapCost, STARTS

from cubeadv.policies.color_policy import RGBNet, Policy
from cubeadv.policies.cnn import PolicyNoStart

import cubeadv.utils as util

    
class Objective:
    """ Objective to be maximized """
    def __init__(self, cfg):
        self.start = STARTS[cfg.start].expand(cfg.opt_batch, -1).to(cfg.device)
        self.goal = STARTS[cfg.goal].expand(cfg.opt_batch, -1).to(cfg.device)
        self.batch = cfg.opt_batch

        self.x0 = torch.cat([self.start, torch.zeros(self.batch, 1).to(cfg.device)], dim=-1)

        self.goal_one_hot = torch.nn.functional.one_hot(torch.tensor([cfg.goal]*cfg.opt_batch), len(STARTS)).view(cfg.opt_batch, -1).to(cfg.device)

        policy_channels_in = 4 if (not cfg.no_depth) else 3

        self.cfg = cfg
        self.device = torch.device(cfg.device)

        if cfg.ngp_field:
            self.policy = PolicyNoStart()
            self.policy.load_state_dict(torch.load(cfg.policy_model_path))
            self.policy = self.policy.to(self.device)
        else:
            net_policy = RGBNet(num_points=cfg.lidar_num_points, channels_in=policy_channels_in)
            net_policy.load_state_dict(torch.load(cfg.policy_model_path))
            net_policy = net_policy.to(self.device)
            self.policy = Policy(net_policy, rgb=True, num_points=cfg.lidar_num_points, height=cfg.lidar_num_channels, channels_in=policy_channels_in)

        if not cfg.single_frame:
            self.init_multiframevars()

    def init(self):
        pass

    def set_device(self, device):
        raise ValueError("This objective function doesn't support changing devices")
        
    def project_to_constraints(self, x):
        return x
        
    def constraint(self, x):
        return torch.tensor(0.) # No constraints

    def init_multiframevars(self):
        # Trajectory params
        self.T = self.cfg.dt * torch.arange(self.cfg.num_steps_traj).to(self.device)

        # Dynamics
        self.dynamics = Dynamics(self.cfg.dt) # Note: 0.001 here is obsolete & unused

        # Cost function for trajectory opt
        # Note: pm.cost does not have a negative sign in front since this is
        # the objective being maximized.

        self.pm = PathMapCost.get_carla_town(self.start.cpu(), self.goal.cpu()).to(self.device)
        if self.cfg.car_start is None:
            x0_ = self.start
            if self.cfg.multistart > 0.1:
                offsets = self.cfg.multistart*(torch.rand(self.batch, 1) * 2 - 1).to(self.device)
                x0_ = self.pm.get_offset(self.start, offsets)
            self.x0 = torch.cat([x0_, self.pm.get_initial_lane_alignment().view(self.batch, -1)], dim=-1)
        else:
            self.x0  = torch.tensor(self.cfg.car_start).view(self.batch, 3).cuda()

        self.cost_fn = lambda x: self.pm.cost(x[:, :2], None)

    def sensor_step(self, x, p, i=0):
        return self.sensor(x, p)

    def policy_step(self, x, p, i=0):
        x_ = x.clone()
        if not self.cfg.dont_detach_yaw:
            x_[:, 2] = x[:, 2].detach()

        if self.cfg.detach:
            o = self.sensor_step(x_.detach(), p, i)
        else:
            o = self.sensor_step(x_, p, i)

        #u = self.policy(o)
        u = self.policy(o, self.goal_one_hot)

        return u

    def F(self, i, t2, t1, x2, x1, p):
        x1 = x1.view(self.batch, -1)
        u = self.policy_step(x1, p, i).view(self.batch)
        xdot = self.dynamics.f(x1.view(self.batch, -1), u)
        x2 = x1 + xdot*(t2 - t1)
        return x2

    def objective_multiframe(self, p, ret_traj=False, reduce=True):
        cost, xs = discrete_adjoint(self.F, self.cost_fn, self.x0, self.T, p, ret_traj=True)
        if reduce:
            cost = cost.mean(0)
        if ret_traj:
            return cost, xs.mean(dim=0)
        else:
            return cost

    def objective(self, p, ret_traj=False):
        raise NotImplementedError()

    def get_random_params(self):
        raise NotImplementedError()

    def get_debug_vars(self, itr, p):
        return None

    def get_constraints(self):
        return torch.tensor(0), torch.tensor(1.)

    def __call__(self, p, ret_traj=False):
        return self.objective(p, ret_traj)

    def render(self, p):
        return None, None

    def render(self, p):
        pass

    def render_pre(self):
        pass

