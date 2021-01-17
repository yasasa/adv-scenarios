import sys, os
sys.path.append("..")
sys.path.append(os.environ["CARLA_PYTHON_PATH"])

import torch
from typing import List
import numpy as np
from cubeadv.fields.ngp import NGPComposeField

from cubeadv.sim.dynamics import Dynamics
from cubeadv.sim.sensors import Camera
from cubeadv.sim.sensors.carla_camera import CarlaCameraCompose
from cubeadv.sim.utils import PathMapCost, STARTS, connect_carla
from cubeadv.fields import NGPField
from cubeadv.utils import normalize, make_functional, set_weights, get_nerf_max, get_nerf_min
from cubeadv.policies.cnn import PolicyNoStart

from configs.parser import arg_parser

from strictfire import StrictFire

from scripts.utils import get_bov_image, get_type_from_module

import objectives

import imageio

def make_gif(path, images, **kwargs):
    imageio.v2.mimsave(path, images, fps=20, **kwargs)

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
        
    print(cost)

    return torch.stack(trajectory), os
    
def set_params(obj, p, keyword):
    param_filter = lambda name: keyword in name
    all_params = torch.tensor([]).cuda()
    meta_list = []
    param_counts = []
        
    params = make_functional(obj.pipeline.nef, param_filter=param_filter, verbose=True)
    current_params = params.param_vector
    set_weights(obj.pipeline.nef, params, 
                current_params + p[:current_params.shape[0]])
    #print('Parameter shape(s):', param_counts)
    return p[current_params.shape[0]:]

def main(config_path: str, parameter_path: str, video: bool = False,
         carla : bool = True, carla_port : int = 2000, 
         device : str = "cuda:0", keyword="codebookcolor", outputfile="standalone_obj_perturb",
         unperturbed: bool = False):
    parser = arg_parser()
    args = parser.parse_args(f"--cfg {config_path}")
    
    start = args.start
    goal = args.goal
    
    policy = PolicyNoStart()
    policy.load_state_dict(torch.load(args.policy_model_path))
    policy = policy.cuda()
    policy = policy.eval()

    pm = PathMapCost.get_carla_town(STARTS[start].view(1, -1), STARTS[goal].view(1, -1))
    
    param_path = parameter_path
    params = torch.from_numpy(np.load(param_path)).cuda()
    obj_params = params
    
    obj_fields = args.obj_fields
    
    transforms = torch.tensor(args.transform_params).cuda()
    if args.objective == "TransformAndColorAttack" and not unperturbed:
        transforms = params[:transforms.shape[0]]
        #print(transforms)
        obj_params = params[transforms.shape[0]:]

    if carla:
        _, world = connect_carla("localhost", carla_port)
        camera = CarlaCameraCompose(world, 200, 66, 100)
        if video:
            viz_cam = CarlaCameraCompose(world, 640, 360, 320)

        for i,field in enumerate(obj_fields):
            obj_field = NGPField(field, scene_scale=torch.ones(3), scene_midpoint=torch.zeros(3))
            if not unperturbed:
                obj_params = set_params(obj_field, obj_params, keyword)
            ob_tr = transforms[4*i: 4*(i+1)]
            camera.add_object(obj_field, ob_tr)
            if video:
                viz_cam.add_object(obj_field, ob_tr)
            
        render_fn = lambda x: camera.read(x)[0]
        if video:
            viz_render_fn = lambda x: viz_cam.read(x)[0]
        else:
            viz_render_fn = None
    else:
        field = NGPComposeField(NGPField(args.ngp_cfg_path))
        field.scene_field.pipeline.nef.ignore_view_dir=False
        
        for i, field_ in enumerate(obj_fields):
            obj_field = NGPField(field_, scene_scale=torch.ones(3), scene_midpoint=torch.zeros(3))
            obj_params = set_params(obj_field, obj_params, keyword)
            ob_tr = transforms[4*i: 4*(i+1)]
            field.add_obj_field(obj_field)
            
        field.set_transform_params(transforms)
        camera = Camera(200, 66, 100)
        if video:
            viz_cam = Camera(640, 360, 320)
        render_fn = lambda x: camera.read(field, x)[0]
        if video:
            viz_render_fn = lambda x: viz_cam.read(field, x)[0]
        else:
            viz_render_fn = None
        
    if args.car_start is None:
        x0 = STARTS[start]
    else:
        x0 = torch.tensor(args.car_start)[:2]

    xs, obs = drive(args, x0, pm, STARTS[goal], start, goal, policy, viz_render_fn, render_fn,  device)
    
    if carla:
        fields = [field for field, _ in camera.obj_fields]
    else:
        fields = field.obj_fields
    fig, ax = get_bov_image(xs, pm, fields=fields, obj_transforms=transforms.view(-1, 4))
    
    suffix = "carla" if carla else "nerf"
    name = config_path.split("/")[-1].rstrip(".txt")
    outputdir = os.path.join(outputfile, name)
    outputdir = os.path.join(outputdir, suffix)
    imagedir = os.path.join(outputdir, "images")
    os.makedirs(outputdir, exist_ok=True)
    os.makedirs(imagedir, exist_ok=True)
    
    bov_plot_path = os.path.join(outputdir, "bov.png")
    video_path = os.path.join(outputdir, "video.gif")
    trajectory_path = os.path.join(outputdir, "trajectory.npy")
    
    np.save(trajectory_path, xs.detach().cpu().numpy())
    fig.savefig(bov_plot_path, bbox_inches='tight')
    
   # make_gif(f"{filename}.gif", os)
    make_gif(video_path, obs)
    
    if video:
        for i, ob in enumerate(obs):
            imageio.imwrite(os.path.join(imagedir, f"{i}.png"), ob)

if __name__ == '__main__':
    sf = StrictFire(main)
