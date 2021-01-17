import os, sys
sys.path.append(os.environ["CARLA_PYTHON_PATH"])
from pathlib import Path
from unittest import TestLoader

import torch
from torch.utils.data import DataLoader, ConcatDataset
from cubeadv.utils import make_functional, set_weights
import numpy as np

import sys

from cubeadv.policies.expert import Expert
from cubeadv.policies.cnn import Policy, PolicyNoStart, PolicyDepth

from cubeadv.sim.sensors import Lidar, Camera 
from cubeadv.sim.sensors.carla_camera import CarlaCameraCompose, CarlaCamera
from cubeadv.fields import NGPField, NGPComposeField
from cubeadv.sim.dynamics import Dynamics
from cubeadv.sim.sensors.carla_camera import BatchCarlaCamera

from cubeadv.sim.utils import PathMapCost, STARTS, connect_carla
from cubeadv.fields.utils import normalize_ngp

import time

from argparse import ArgumentParser

from scripts.datastore import Datastore, DatastoreDataset

sys.path.append("..")
from configs.parser import arg_parser as extra_arg_parser


def setup_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def setup_output(path):
    os.makedirs(path, exist_ok=True)

def arg_parser(parser):
    parser.add_argument("--carla", action='store_true', default=False)
    parser.add_argument("--carla-batch", type=int, default=0)
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--input-path", type=str, default=None)

    parser.add_argument("--extra-db-dir", type=str, default=None)
    parser.add_argument("--load-extra-databases", action='store_true')
    parser.add_argument("--experiment-config", type=str, default=None)
    parser.add_argument("--parameter-dir", type=str, default=None)
    
    parser.add_argument("--priority-sample", action='store_true')

    parser.add_argument("--collect-data", action='store_true', default=False)
    parser.add_argument("--fit-policy", action='store_true', default=False)

    parser.add_argument("--no-start", action='store_true', default=True)
    parser.add_argument("--depth", action='store_true', default=False)

    parser.add_argument("--rollout-batch-size", type=int, default=6)
    parser.add_argument("--uniform", action='store_true', default=False, help="Ignores rollout batch size to uniformly get trajectory classes")
    parser.add_argument("--train-samples-per-epoch", type=int, default=1)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fit-iter", type=int, default=200)
    parser.add_argument("--valid-iter", type=int, default=10)

    parser.add_argument("--datastore-capacity", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--dataset-num-workers", type=int, default=-1)

    parser.add_argument("--device", default="cuda:0", type=str)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nerf-cfg", type=str, default="wisp/configs/ngp_nerf_bg_new.yaml")

    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--T", type=int, default=1000)
    return parser

def sample_start_goal(batch_size, uniform=False, priority_sample=None):
    if priority_sample is not None:
        s, g = priority_sample
        l1 = torch.ones(batch_size).long() * s
        l2 = torch.ones(batch_size).long() * g
        idx = torch.stack([l1, l2], dim=-1)
    elif uniform:
        idx = torch.cartesian_prod(torch.arange(3), torch.arange(3))
        idx = idx[idx[:, 0] != idx[:, 1]]
        print(idx)
    else:
        idx = torch.randint(2, size=(batch_size, 2))
    print(idx)

    start, goal = STARTS[idx[:, 0]], STARTS[idx[:, 1]]
    map = PathMapCost.get_carla_town(start, goal)
    goal_one_hot = torch.nn.functional.one_hot(idx[:, 1], num_classes=len(STARTS))
    start_one_hot = torch.nn.functional.one_hot(idx[:, 0], num_classes=len(STARTS))

    return map, start, goal, start_one_hot, goal_one_hot

@torch.no_grad()
def sample_trajectory(args, render_fn, dynamics, datastore, uniform=False, noise=0.0):
    priority_sample=None
    if args.priority_sample:
        parser = extra_arg_parser()
        eargs = parser.parse_args(f"--cfg {args.experiment_config}")
        priority_sample = (eargs.start, eargs.goal)
        
        
    map, start, goals, starts_one_hot, goals_one_hot = sample_start_goal(args.rollout_batch_size, uniform, priority_sample)
    map = map.to(args.device)
    start = start.to(args.device)

    thetas = map.get_initial_lane_alignment()

    start_noise = torch.from_numpy(np.clip(np.random.randn(args.rollout_batch_size), a_max=4., a_min=-4.)[..., None]).to(args.device).float()
    if args.priority_sample:
        start = torch.tensor(eargs.car_start).view(1, -1).expand(args.rollout_batch_size, -1).to(args.device)[..., :2]
        
    random_start = map.get_offset(start, start_noise)
    x0 = torch.cat([random_start, thetas.view(-1, 1)], dim=-1)
    x0 = x0.to(args.device)
    expert = Expert(-2., map)
    expert.load_params(1., 20., 50.)

    trajectory = []
    observations = []
    expert_controls = []

    prep = lambda x: x.detach().cpu().numpy()

    x = x0
    for t in range(args.T):
        o, d = render_fn(x)

        u_e = expert(x[:, :2])
        u = u_e + torch.randn_like(u_e) * noise 
        x = dynamics(x, u)

        trajectory.append(x)
        observations.append(o)
        expert_controls.append(u_e)

        datastore.write_batch(prep(u_e), [prep(x), prep(goals_one_hot), prep(o), prep(d), prep(starts_one_hot)])

    return torch.stack(trajectory), torch.stack(observations), torch.stack(expert_controls)

def one_hot_to_coord(onehot, device):
    id = onehot.argmax(dim=-1)
    gt = STARTS[id].to(device)
    gt = normalize_ngp(gt, 1.)
    return gt

def unpack(data, device):
    if len(data) > 5: # includes depth
        controls, x, goal, image, depth, start = data
    else:
        controls, x, goal, image, start = data
        depth = torch.zeros(*image.shape[:-1]).type_as(image)

    # UNUSED
#    goal_gt = one_hot_to_coord(goal, device)
#    start_gt = one_hot_to_coord(start, device)

  #  x = x.to(device)
  #  x = normalize_ngp(x, 1.)

    # USED
    controls = controls.to(device)
    goal = goal.to(device)
    start = start.to(device)

    image = image.to(device)
    if image.dtype != torch.float32:
        image = image.float()
        image = image / 255.
    image = image.permute(0, 3, 1, 2) # BCHW

    depth = depth.to(device)
    depth = depth.view(-1, 1, image.shape[-2], image.shape[-1])

    return controls, goal, image, depth, start

@torch.no_grad()
def validate(policy, testset, device):
    policy = policy.eval()
    total_loss = 0
    for _, ld in enumerate(testset):
        controls, goal, image, depth, start = unpack(ld, device)
        u = policy(image, goal, depth, start)
        loss = torch.nn.functional.mse_loss(controls, u.squeeze())
        total_loss += loss.item()
    policy.train()

    return total_loss / len(testset)


def fit_policy(args, policy, optim, trainset, testset, device):
    for i in range(args.fit_iter):
        for _, ld in enumerate(trainset):
            controls, goal, image, depth, start = unpack(ld, device)
            optim.zero_grad()
            u = policy(image, goal, depth, start)
            loss = torch.nn.functional.mse_loss(controls, u.squeeze())
            loss.backward()
            optim.step()

        if i % args.valid_iter == args.valid_iter - 1:
            valid_loss = validate(policy, testset, device)
            suff = "_no_start" if args.no_start else ""
            torch.save(policy.state_dict(), Path(args.output_path, f"policy_epoch_{i}{suff}.pt"))
            print(f"[Epoch {i+1}] Loss: {loss.item() : .7f} Validation: {valid_loss: .7f}")

def load_params(args, eargs, fields):
    param_dir = args.parameter_dir
    if param_dir is None:
        param_dir = os.path.join(eargs.output_dir, "run-0", "parameters-log-50.npy")
    p = torch.from_numpy(np.load(param_dir)).cuda()

    for field in fields:
        param_filter = lambda name: eargs.param_search_keyword in name
        params = make_functional(field.pipeline.nef, param_filter=param_filter, verbose=True)
        current_params = params.param_vector
        set_weights(field.pipeline.nef, params,
                    current_params + p[:current_params.shape[0]])
        p = p[current_params.shape[0]:]

def add_compose_objects(args, field):
    parser = extra_arg_parser()
    eargs = parser.parse_args(f"--cfg {args.experiment_config}")
    transforms = torch.tensor(eargs.transform_params)
#   transforms_min = torch.tensor(eargs.transform_min)
#   transforms_max = torch.tensor(eargs.transform_max)
#   transforms = torch.rand_like(transforms_min) * (transforms_max - transforms_min) + transforms_min
    transforms = transforms.split(4)

    fields = []
    for i, obj in enumerate(eargs.obj_fields):
        obj = NGPField(obj, scene_midpoint=torch.zeros(3), scene_scale=torch.ones(3))
        fields.append(obj)
        field.add_obj_field(obj, transforms[i].cuda())

 #   load_params(args, eargs, fields)
    extra = os.path.splitext(os.path.basename(args.experiment_config))[0]
    return f"extra_train_db_{extra}"


def create_compose_sensor(args):
    test_db = None
    if args.carla:
        _, world = connect_carla("localhost", args.port)
        field = CarlaCameraCompose(world, 200, 66, 100, box=False)
        render_fn = field.read
        def render_fn(x):
            o, d = field.read(x)
            o = (o*255).long()
            return o, d
    else:
        field = NGPComposeField(NGPField(args.nerf_cfg))
        camera = Camera(200, 66, 100)
        def render_fn(x):
            o, d = camera.read(field, x)
            o = (o*255).long()
            return o, d

    train_db = add_compose_objects(args, field)
    print(train_db)
    return render_fn, train_db, test_db

def create_sensor(args):
    if args.carla:
        _, world = connect_carla("localhost", args.port)
        camera = CarlaCamera(world, 200, 66, 100)
        render_fn = camera.read
        train_db = "training_db"
        test_db = "testing_db"
    else:
        field = NGPField(args.nerf_cfg)
        field.pipeline.nef.ignore_view_dir = False
        camera = Camera(200, 66, 100)
        def render_fn(x):
            o, d = camera.read(field, x)
            o = (o*255).long()
            return o, d
        train_db = "nerf_training_db"
        test_db = "nerf_testing_db"
        
    return render_fn, train_db, test_db

def collect_data(args):
    if args.experiment_config is None:
        render_fn, train_db, test_db = create_sensor(args)
    else:
        render_fn, train_db, test_db = create_compose_sensor(args)

    dynamics = Dynamics(args.dt)
    
    train_datastore = Datastore(args.datastore_capacity, Path(args.output_path, train_db))
    
    if test_db:
        test_datastore = Datastore(args.datastore_capacity, Path(args.output_path, test_db))
        sample_trajectory(args, render_fn, dynamics, test_datastore, uniform=True)

    for i in range(args.train_samples_per_epoch):
        print(f"Collecting {i}th training trajectory with {args.rollout_batch_size} rollouts")
        sample_trajectory(args, render_fn, dynamics, train_datastore, uniform=args.uniform, noise=0)

    
    train_datastore.sync()
    if test_db:
        test_datastore.sync()

def load_datastore(args, train_db, test_db):
    train_datastore = Datastore(args.datastore_capacity, Path(args.input_path, train_db))
    test_datastore = Datastore(args.datastore_capacity, Path(args.input_path, test_db))
    return train_datastore, test_datastore


if __name__ == "__main__":
    parser = arg_parser(ArgumentParser())
    args = parser.parse_args()

    if args.input_path is None:
        args.input_path = args.output_path
    if args.extra_db_dir is None:
        args.extra_db_dir = args.input_path
        
    setup_seeds(args.seed)
    setup_output(args.output_path)

    if args.collect_data:
        collect_data(args)

    if args.fit_policy:
        device = torch.device(args.device)
        if args.no_start:
            if args.depth:
                policy = PolicyDepth().to(device)
            else:
                policy = PolicyNoStart().to(device)
        else:
            policy = Policy().to(device)
        optimizer = torch.optim.Adam(policy.parameters())

        carla_train_datastore, carla_test_datastore = load_datastore(args,"training_db", "testing_db")
        print(carla_train_datastore.size)
        nerf_train_datastore, nerf_test_datastore = load_datastore(args, "nerf_training_db", "nerf_testing_db")
        print(nerf_train_datastore.size)
        
        if nerf_train_datastore.size > 0 and not args.carla:
            print("Training on both nerf and carla data")
            train_dataset = ConcatDataset([DatastoreDataset(nerf_train_datastore), DatastoreDataset(carla_train_datastore)])
        else:
            train_dataset = DatastoreDataset(carla_train_datastore)

        if nerf_test_datastore.size > 0:
            test_dataset = DatastoreDataset(nerf_test_datastore)
        else:
            test_dataset = DatastoreDataset(carla_test_datastore)

        if args.load_extra_databases:
            files = [os.path.join(args.extra_db_dir, p)  for p in os.listdir(args.extra_db_dir) if p.startswith("extra_train_db") and not ";afjig;oijdfslgkdjs" in p]
            datastores = []
            for f in files:
                print(f)
                store = DatastoreDataset(Datastore(-1, f))
           #     if len(store) > 8000:
           #         store = torch.utils.data.Subset(store, torch.arange(8000))
                
                datastores.append(store)
                print(len(store))
                
            
            print(f"Found {len(files)} extra databases to load")
            train_dataset = ConcatDataset([train_dataset] + datastores)
            test_dataset = ConcatDataset(datastores)
            
        test_dataset = DatastoreDataset(Datastore(-1, os.path.join(args.extra_db_dir, "extra_train_db_NGP-TransformAttack-1-2-2car-3hydrant-earlier-2")))

        trainloader = DataLoader(train_dataset, args.batch_size, True, num_workers=args.dataset_num_workers)
        testloader = DataLoader(test_dataset, args.test_batch_size, True, num_workers=args.dataset_num_workers)

        fit_policy(args, policy, optimizer, trainloader, testloader, device)
