
import torch
from typing import List
import numpy as np
from cubeadv.fields.ngp import NGPComposeField

from cubeadv.sim.dynamics import Dynamics
from cubeadv.sim.sensors import Camera
from cubeadv.sim.utils import PathMapCost, STARTS
from cubeadv.fields import NGPField
from cubeadv.utils import normalize, make_functional, set_weights, get_nerf_max, get_nerf_min
from cubeadv.policies.cnn import PolicyNoStart

from configs.parser import arg_parser

from strictfire import StrictFire
from scripts.objectives.ngp_objectives import NGPTransformAttack

from scripts.utils import get_bov_image, get_type_from_module

import objectives
import os

import imageio

from matplotlib import pyplot as plt


@torch.no_grad()
def drive(cfg, start, pm, goal, start_idx, goal_idx, policy, visual_render_fn, render_fn, device):
    pm = pm.to(device)
    trajectory = []
    os = []
    dynamics = Dynamics(0.01)

    goal_one_hot = torch.nn.functional.one_hot(torch.tensor([goal_idx]), len(STARTS)).cuda()
    start_one_hot = torch.nn.functional.one_hot(torch.tensor([start_idx]), len(STARTS)).cuda()

    x = torch.cat([start.cuda(), pm.get_initial_lane_alignment()], dim=-1)
    x[:2] = pm.get_offset(x[:2], 0.)
    x = x.cuda()
    T = torch.arange(cfg.num_steps_traj)
    cost = 0
    for t in T:
        o = render_fn(x.view(1, -1))
        if visual_render_fn is None:
            o_viz = o
        else:
            o_viz = visual_render_fn(x.view(1, -1))
        o_viz = (o_viz * 255).byte()
        o = o.permute(0, 3, 1, 2)
        u = policy(o, goal_one_hot, start_one_hot).view(-1)
        x = dynamics(x, u).squeeze()
        cost += pm.cost(x[:2], None)
     #   print(t, x)
        trajectory.append(torch.cat([x.detach().cpu().squeeze(), u.detach().cpu()]))
        os.append(o_viz[0].detach().cpu().numpy())
        
    return cost
    
def set_params(obj, p, keyword):
    param_filter = lambda name: keyword in name
    all_params = torch.tensor([]).cuda()
    meta_list = []
    param_counts = []
        
    params = make_functional(obj.pipeline.nef, param_filter=param_filter, verbose=True)
    current_params = params.param_vector
    print(current_params.shape)
    set_weights(obj.pipeline.nef, params, 
                current_params + p[:current_params.shape[0]])
    print('Parameter shape(s):', param_counts)
    return p[current_params.shape[0]:]

def main(config_path: str, output_name: str, output_path: str = "."):
    parser = arg_parser()
    args = parser.parse_args(f"--cfg {config_path}")
    
    start = args.start
    goal = args.goal
    
    policy = PolicyNoStart()
    policy.load_state_dict(torch.load(args.policy_model_path))
    policy = policy.cuda()
    policy = policy.eval()

    pm = PathMapCost.get_carla_town(STARTS[start].view(1, -1), STARTS[goal].view(1, -1))
    
    objective = NGPTransformAttack(args)
    p0 = torch.tensor(args.transform_params)
    
    for i in [1, 5, 9, 13, 17]:
        plt.clf()
        plt.cla()
        x, v = objective.plot_loss(p0, 200, i)
        x = x.cpu()
        v = v.cpu()
        
        np.save(os.path.join(output_path, f"{output_name}_xs_{i}.npy"), x.numpy())
        np.save(os.path.join(output_path, f"{output_name}_values_{i}.npy"), v.numpy())
        
        plt.plot(x, v)
        plt.savefig(f"{output_name}_{i}.png")
    

if __name__ == '__main__':
    sf = StrictFire(main)
