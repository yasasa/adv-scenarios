import torch
import numpy as np

from cubeadv.sim.dynamics import Dynamics
from cubeadv.sim.utils import PathMapCost, STARTS, connect_carla
from cubeadv.fields import NGPField
from cubeadv.sim.sensors import CarlaCamera, Camera
from cubeadv.policies.cnn import PolicyNoStart

from strictfire import StrictFire

from scripts.utils import get_bov_image, get_type_from_module

import objectives

import imageio

def make_gif(path, images):
    imageio.v2.mimwrite(path, images, fps=10)

@torch.no_grad()
def drive(start, pm, goal, start_idx, goal_idx, policy, render_fn, device):
    pm = pm.to(device)
    trajectory = []
    os = []
    dynamics = Dynamics(0.01)

    goal_one_hot = torch.nn.functional.one_hot(torch.tensor([goal_idx]), len(STARTS)).cuda()
    start_one_hot = torch.nn.functional.one_hot(torch.tensor([start_idx]), len(STARTS)).cuda()

    x = torch.cat([start.cuda(), pm.get_initial_lane_alignment()], dim=-1)
    x[:2] = pm.get_offset(x[:2], 0.)
    x = x.cuda()
    T = torch.arange(1000)
    for t in T:
        o = render_fn(x.view(1, -1))
        o = o.permute(0, 3, 1, 2).cuda()

        u = policy(o, goal_one_hot, start_one_hot).view(-1)
        x = dynamics(x, u).squeeze()
        trajectory.append(x.detach().cpu())
        os.append(o[0].permute(1, 2, 0).detach().cpu().numpy())

    return torch.stack(trajectory), os


def main(start : int, goal: int, policy_path : str, param_path : str, objective: str = "NGPFieldAttack", output_path: str = ".", carla : bool = True, carla_port : int = 2000, device : str = "cuda:0"):
    policy = PolicyNoStart()
    policy.load_state_dict(torch.load(policy_path))
    policy = policy.cuda()
    policy = policy.eval()

    pm = PathMapCost.get_carla_town(STARTS[start].view(1, -1), STARTS[goal].view(1, -1))
    objective = get_type_from_module(objective, objectives)

    params = np.load(param_path)[0]
    params = torch.from_numpy(params).to(device).permute(1, 2, 0)

    if carla:
        _, world = connect_carla("localhost", carla_port)
        camera = CarlaCamera(world, 200, 66, 100)
        render_fn = lambda x: camera.read(x, convert=True)[0].to(device) + params
    else:
        field = NGPField("wisp/configs/ngp_nerf.yaml", True)
        camera = Camera(200, 66, 100)
        render_fn = lambda x: camera.read(field, x)[0] + params

    xs, os = drive(STARTS[start], pm, STARTS[goal], start, goal, policy, render_fn, device)

    fig, ax = get_bov_image(xs, pm)
    
    suffix = "carla" if carla else "nerf"
    filename = f"standalone_pixel_perturb_{suffix}"
    fig.savefig(f"{filename}.png")
    
    make_gif(f"{filename}.gif", os)

if __name__ == '__main__':
    sf = StrictFire(main)