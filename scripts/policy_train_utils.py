from distutils.command.build import build
import torch
import imageio
import numpy as np
from cubeadv.policies.expert import Expert
from cubeadv.policies.color_policy import Policy, RGBNet
from cubeadv.sim.sensors import *

from cubeadv.sim.dynamics import Dynamics

from cubeadv.sim.sensors import Camera, Lidar

from cubeadv.sim.utils import PathMapCost

from cubeadv.fields import NGPField, NGPComposeField
import cubeadv.utils as util

from copy import deepcopy

import matplotlib.pyplot as plt

def build_nerf_lidar(cfg):
    from cubeadv.fields.base_rf import MockRF
    if cfg.mock:
        field = MockRF(cfg.num_obj)
    else:
        if cfg.my580:
            scale = util.MY580_SCALE
            mid = util.MY580_MIDPOINT
            transform = util.get_permutation(torch.tensor([0, 2, 1]), homog=True)
            transform[:, [1, 2]] *= -1
            field = NGPField(cfg.ngp_cfg_path, scale, mid, transform)
        else:
            field = NGPField(cfg.ngp_cfg_path)

        if cfg.obj_cfg_path or cfg.obj_fields:
            field = NGPComposeField(field)

        
    if cfg.camera:
        sensor = Camera(cfg.cam_width, cfg.cam_height, cfg.cam_focal)
    else:
        sensor = Lidar(cfg.beam_width, cfg.beam_height)
        return field, sensor
        
    return field, sensor

def setup_bev_plot(fig, ax):
    ax.xlim(80, 110)
    ax.ylim(140, 110)
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    ax.legend()


def plot_road_bg2(fig, ax):
    segments = PathMapCost.get_all_lanes()
    
    for segment in segments:
        ax.plot(segment[:, 0], segment[:, 1], color='black', label="Road Edges")

    return segments

def plot_road_bg(fig, ax):
    LANE_WIDTH=4

    path_segments = np.array([[129.4, 132.5], [96.4, 132.5], [93.4, 130.7], [91.2, 128.5], [90.4, 125.5], [90.4, 101.]])

    path_vectors = path_segments[1:] - path_segments[:-1]
    segment_lengths = np.linalg.norm(path_vectors, axis=-1)
    path_vectors_normalized = path_vectors / segment_lengths.reshape(-1, 1)

    rot_path_vectors = path_vectors_normalized.dot(np.array([[0, -1],[1, 0]]))
    top_road_segments = np.copy(path_segments)
    top_road_segments[1:] -= rot_path_vectors * LANE_WIDTH/2
    top_road_segments[0] -= rot_path_vectors[0] * LANE_WIDTH / 2

    midpoint_vector = np.array([0, 132.5])
    bot_road_segments = (top_road_segments - midpoint_vector).dot(np.array([[1, 0],[0, -1]])) + midpoint_vector

    def segments_to_lines(seg):
        return seg[1:] - seg[:-1]

    top_road_lines = segments_to_lines(top_road_segments)
    bot_road_lines = segments_to_lines(bot_road_segments)
    side_road_segment = np.array([[90.4 - LANE_WIDTH/2, 101], [90.4 - LANE_WIDTH/2, 162 + LANE_WIDTH]])
    
    xmin=90.4 - 15.
    xmax=90.4 + 15.
    ymin=131.5 - 15.
    ymax=131.5 + 15.
    
    im = imageio.imread("resources/carla_bg.png")
    ax.imshow(im,extent=(xmin, xmax, ymax, ymin))

    # Plot data roads
    ax.plot(side_road_segment[:, 0], side_road_segment[:, 1], color='green', label="Road Edges", alpha=0.2, linestyle='--')
    ax.plot(top_road_segments[:, 0], top_road_segments[:, 1], color='green', alpha=0.2, linestyle='--')
    ax.plot(bot_road_segments[:, 0], bot_road_segments[:, 1], color='green', alpha=0.2, linestyle='--')
    plt.xlim(xmin, xmax)
    plt.ylim(ymax, ymin)

def get_high_resolution_sensor(cfg, width, height):
    cfg = deepcopy(cfg)
    cfg.lidar_num_points = width * height
    cfg.lidar_num_channels = height
    return build_nerf_lidar(cfg)

def get_available_memory():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved()
    a = torch.cuda.memory_allocated()
    f = a / int(1e6)  # free inside reserved
    return f

@torch.no_grad()
def sample_policy_trajectory(cfg,
                             policy: Policy,
                             x0: torch.Tensor,
                             T = 200,
                             dt = 1./200):
    lidar = build_nerf_lidar(cfg)

    dynamics = Dynamics(dt)
    runner = Runner(lidar, dynamics, policy, lambda x, u: 0)

    _, xs = runner.run_steps(x0, None, T, dt)
    return xs.detach().cpu()

@torch.no_grad()
def get_camera_images(cfg,
                     policy: Policy,
                     sensor,
                     xs: torch.Tensor,
                     dt = 1./200):
    print(get_available_memory())
    dynamics = Dynamics(dt)
    runner = Runner(None, dynamics, policy, lambda x, u: 0)
    runner.camera_runner(sensor, xs, None, "experiments/policy-images")

def plot_policy_trajectory(x0, cfg):
    print(get_available_memory())
    policy_channels_in = 4 if (not cfg.no_depth) else 3
    net_policy = RGBNet(num_points=cfg.lidar_num_points, channels_in=policy_channels_in)
    net_policy.load_state_dict(torch.load(cfg.policy_model_path))
    net_policy.cuda()

    policy = Policy(net_policy, True, cfg.lidar_num_points, cfg.lidar_num_channels, policy_channels_in)

    xs = sample_policy_trajectory(cfg, policy, x0, cfg.num_steps_traj, 1./cfg.num_steps_traj)
    torch.cuda.empty_cache()
    print(get_available_memory())
    get_camera_images(cfg, policy, get_high_resolution_sensor(cfg, 180, 45), xs, 1./cfg.num_steps_traj)

    fig, ax = plt.subplots()
    plot_road_bg(fig, ax)
    ax.plot(xs[:, 0], xs[:, 1], label="Policy output")
    setup_bev_plot(fig, ax)

    return fig, ax




