import configargparse

def arg_parser():
    parser = configargparse.ArgParser(default_config_files=['../configs/custom_train_configs/default_params.txt'])

    parser.add('--cfg', required=True, is_config_file=True, help='config file path')
    
    parser.add_argument("--exp_name", default=None, type=str)

    parser.add_argument("--output_dir", default="experiments",
                        type=str,help='output file path')

    parser.add_argument("--save_params", default=0,
                        type=int,help='save params every save_params th iteration, not saving params if 0')

    parser.add_argument('--nerf_config', type=str, help='Location of the nerf config parameters')

    parser.add_argument('--policy_model_path', type=str, help='Location of the nerf config parameters')

    parser.add_argument('--debug_var_freq', type=int, default=1, help='Frequency at which debug variables are logged')

    parser.add_argument("--save_param_traj", action='store_true', help='whether to save optimization params every iteration')
    
    parser.add_argument("--process_count", type=int, default=0, help="Total number of processes to use, if 0 run in the main process")

    parser.add_argument("--manual_log", action='store_true', help='Whether to manually log stats instead of using tensorboard')

    # --------- Setup --------------
    parser.add_argument('--goal', type=int, default=1)
    parser.add_argument('--start', type=int, default=0)

    parser.add_argument('--car-start', type=float, nargs=3, default=None)
    
    parser.add_argument('--single_frame', action='store_true', help='Optimize on a single frame if true, otherwise optimize on a trajectory')
    parser.add_argument('--dont_detach_yaw', action='store_true',help='Dont detach yaw grad if true')

    parser.add_argument('--cube', action='store_true',help='Cube parameter optimization')
    parser.add_argument('--num_cubes', type=int, default=5, help='Num cubes in cube parameter optimization')
    parser.add_argument('--num_steps_traj', type=int, default=50, help='Num steps in trajectory optimization')
    parser.add_argument('--tn_traj', type=float, default=1.0, help='duration of trajectory')
    parser.add_argument('--perturb', action='store_true',help='Perturbation optimization')
    parser.add_argument('--ref_perturb', action='store_true',help='Perturbation ontop of reference optimization')
    parser.add_argument('--nerf_texture', action='store_true',help='Perturbation ontop of reference optimization')
    parser.add_argument('--objective', type=str, help='Class name of the objective function to use')
    
    
    parser.add_argument('--detach', action='store_true', default=False, help='Detach state gradients from simulator')


    # --------- Plenoxels --------------
    parser.add_argument('--plenoxels', action='store_true',help='Use plenoxels model if true')
    parser.add_argument('--plenoxel_ckpt_path', type=str, default=None, help='Plenoxel model checkpoint')
    parser.add_argument('--plenoxel_cfg_path', type=str, default=None, help='Plenoxel cfg path')
    parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')

    # --------- InstantNGP --------------
    parser.add_argument('--ngp', action='store_true',help='Use instant ngp model if true')
    parser.add_argument('--ngp-field', action='store_true')
    parser.add_argument('--mock', action='store_true')
    
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--ngp_cfg_path', type=str, default=None, help='Instant ngp config path')
    parser.add_argument('--obj_cfg_path', type=str, default=None, help='Instant ngp config path')
    parser.add_argument('--obj_fields', type=str, nargs='*', default=[], help="List of object fields")
    parser.add_argument('--compose_transform_attack', action='store_true', help='Adversarial attack on object composition parameters')
    parser.add_argument('--compose_colour_attack', action='store_true', help='Adversarial attack on object composition colour parameters')
    parser.add_argument('--num_obj', type=int, default=1, help='number of objects to insert into scene')
    parser.add_argument("--box", default=False, action='store_true', help="Whether to use box object or engine object")
    parser.add_argument("--my580", default=False, action='store_true', help="Whether to use box object or engine object")

    parser.add_argument('--camera', action='store_true', default=False)
    parser.add_argument('--cam-width', type=int, default=200)
    parser.add_argument('--cam-height', type=int, default=66)
    parser.add_argument('--cam-focal', type=int, default=100)

    # --------- Common Instant NGP + Plenoxels ---------
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--no_depth', action='store_true', help='No depth channel in sensor output')
    parser.add_argument('--lidar_num_points', type=int, default=3200, help='Total num points in lidar scan')
    parser.add_argument('--lidar_num_channels', type=int, default=32, help='Height of lidar scan')
    parser.add_argument('--voxel_attack', action='store_true', help='Adversarial attack on voxel parameters')

    # -------- Initial Params ---------
    parser.add_argument('--opt-batch', type=int, default=1)
    parser.add_argument('--multistart', type=float, default=0.)
    parser.add_argument('--init_params', type=str, default=None, help='Location of the initial parameters')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--num_rand_inits', type=int, default=5, help='Random seed')

    # -------- Loss  -----------
    parser.add_argument('--constraint', type=str,
            help='constraint type; valid values are [none, barrier, potential]')

    parser.add_argument('--mu', type=float,
            help='scaling factor for log barrier constraint')

    # -------- Special params for nerf_texture ---------
    parser.add_argument('--cube_param_path', type=str, default=None, help='Location of the cube parameters')
    
    # -------- Special params for object attacks ---------
    parser.add_argument("--transform-params", type=float, nargs="+",
                          help="Parameters for the object tranform")
                          
    parser.add_argument("--transform-min", type=float, nargs="+",
                          help="Lower bound on parameters for the object tranform")
    parser.add_argument("--transform-max", type=float, nargs="+",
                          help="Upper bound on parameters for the object tranform")
                          
                          
    parser.add_argument("--param-search-keyword", type=str, default='decoder_color',
                          help="Keyword to find specific params")

    # -------- Optimizer -----------
    parser.add_argument('--iterations', type=int, help='iterations')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--momentum', type=float, help='momentum')
    parser.add_argument('--scheduling', type=float, help='learning rate scheduling')
    parser.add_argument('--grad_clipping', type=float, help='gradient clipping')
    parser.add_argument('--optimizer', type=str, help='Optimizer algo')
    
    parser.add_argument('--device', type=str, default="cuda:0")

    return parser
