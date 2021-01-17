import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
import matplotlib.patches as patches
from cubeadv.utils import normalize, denormalize, get_nerf_max, get_nerf_min
import torch
import matplotlib.transforms as transforms
from configs.parser import arg_parser
from cubeadv.policies.expert import Expert
from policy_train_utils import plot_road_bg

# ================================ Arg Parser =================================

#parser = argparse.ArgumentParser(description='Plot results.')
#parser.add_argument("--data_folder",
#                    default="out_tmp",
#                    type=str,help='data folder path')
#parser.add_argument('--straight_traj', action='store_true',help='Optimize straight trajectory (instead of turn) if true')

# ============================== Training Curves ==============================

def plot_training_curves(data_folder, show_params=False):
    if show_params:
        f = plt.figure(figsize=(18,6))
        num_plots = 3
    else:
        f = plt.figure(figsize=(12,6))
        num_plots = 2
    ax1 = plt.subplot(1, num_plots, 1)
    ax1.set_title("Training curves for all runs")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Total loss (objective loss + constraint loss)");
    ax2 = plt.subplot(1, num_plots, 2)
    ax2.set_title("Objective losses for all runs")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Objective loss")

    if show_params:
        ax3 = plt.subplot(1, num_plots, 3)
        ax3.set_title("Parameter values for all runs")
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Parameter value")

    best_losses = []

    for subdir, dirs, files in os.walk(data_folder):
        if len(subdir.rsplit('/', 1)) < 2 or subdir.rsplit('/', 1)[1][0:3] != 'run':
            continue
        obj_losses = np.load(os.path.join(subdir, 'obj_loss-log.npy'))
        constr_losses = np.load(os.path.join(subdir, 'constr_loss-log.npy'))
        total_losses = obj_losses + constr_losses

        ax1.plot(np.arange(total_losses.shape[0]), total_losses)
        ax2.plot(np.arange(total_losses.shape[0]), obj_losses)

        if show_params:
            params_per_iter = np.load(os.path.join(subdir, 'params-log.npy'))
            ax3.plot(np.arange(total_losses.shape[0]), params_per_iter)

        best_feasible_loss = min(total_losses[np.where(constr_losses < 10**(-4))])
        best_losses.append(best_feasible_loss)

    print('Best loss across all runs:', min(best_losses))
    print('Avg. best loss across all runs:', np.mean(best_losses))
    print('Std. dev of best loss across all runs:', np.std(best_losses))

    plot_name = 'results_training.png'
    f.savefig(plot_name, bbox_inches ='tight')
    print('training curve plots saved to: {}'.format(plot_name))

# ============================ Cubes & Trajectories ===========================

@torch.no_grad()
def run_expert(x0, T, dynamics, expert, pm):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    expert.reset()

    x = torch.from_numpy(x0).to(device)
    save_x = []
    save_prj = []
    save_lanes = []

    cost = 0.

    for i in range(T):
        ue = expert(x.cpu().numpy())
        u = torch.from_numpy(ue).type_as(x).squeeze()
        #print(u)

        save_x.append(x)
        x = dynamics(x, u).float()
        cost += pm.cost(x, u)

        #step_cost, _, lane_id, road_prj = pm.policy_cost(x, u)
        #print(x, step_cost)
        #cost += step_cost

        #save_prj.append(road_prj)
        #save_lanes.append(lane_id)

    print('Expert cost:', cost)

    return save_x

# 2d birds-eye-view plot
def plot_bov_traj(cfg, cube_params, params_normalized, num_cube_params=12, cube_num=5, debug_use_init_params=False):

    fig, ax  = plt.subplots()
    plt.style.use('seaborn-whitegrid')

    transform_params = None
    if cfg.objective in ['MultiFrameNGPColourAttack']:
        transform_params = torch.load(os.path.join(cfg.output_dir, 'run-{}'.format(run_idx), 'transform_params-log.pt'))


    # Plot roads
    plot_road_bg(fig, ax)

    # Plot cube trajectory
    if cfg.compose_colour_attack:
        objective_type = objectives.MultiFrameNGPColourAttack
    elif cfg.compose_transform_attack:
        objective_type = objectives.MultiFrameCompose
    elif cfg.voxel_attack:
        objective_type = objectives.MultiFrameVoxel
    elif cfg.perturb:
        objective_type = objectives.MultiFramePerturb
    else:
        objective_type = objectives.MultiFrameCube
    if transform_params is None:
        objective = objective_type(cfg)
    else:
        objective = objective_type(cfg, transform_params)
    with torch.no_grad():
        if debug_use_init_params:
            if cfg.perturb:
                params_normalized = torch.zeros_like(params_normalized) 
            elif cfg.compose_colour_attack:
                params_normalized = objective.original_params
            else:
                raise ValueError
        cost, tr1 = objective(params_normalized.cuda(), ret_traj=True)
    print('Cost with adv-attack:', cost)
    for i in range(len(tr1)):
        tr1[i] = tr1[i].cpu().detach().numpy()
    tr1 = np.array(tr1)
    ax.plot(tr1[:, 0], tr1[:, 1], color='orange', label="With Adv-Attack Trajectory")

    # Plot ref trajectory
    if cfg.straight_traj:
        path_segments = np.array([[92.4, 111], [92.4, 124]])
        ax.plot(path_segments[:, 0], path_segments[:, 1], linestyle='--', label = "Nominal trajectory for the car")
    else:
        target = [0, 0, 0, 0, 0]
        expert = Expert(target, objective.pm)
        expert.load_params(1.5, 0, 165.)
        te = run_expert(objective.x0.cpu().numpy(), cfg.num_steps_traj, objective.dynamics, expert, objective.pm)
        te = torch.stack(te).cpu().numpy()
        ax.plot(te[:, 0], te[:, 1], color='green', linestyle='--', label="Expert Trajectory")

    # Plot initial cube locations
    #x = np.reshape(cubes_init.cpu(), (-1, num_cube_params))[:, 0]
    #y = np.reshape(cubes_init.cpu(), (-1, num_cube_params))[:, 1]
    #plt.scatter(x, y, color='red')

    # Plot final cube locations
    '''
    x = np.reshape(cube_params, (-1, num_cube_params))[:, 0]
    y = np.reshape(cube_params, (-1, num_cube_params))[:, 1]
    plt.scatter(x, y, color='green')
    final_pos = np.reshape(cube_params, (-1, num_cube_params))
    for i in range(cube_num):
        for j in range(3,6):
            if final_pos[i][j] > 1.0:
                print('WARNING, colour constraint violated, clamping')
                print(final_pos[i])
                final_pos[i][j] = 1.0
        x, y = final_pos[i][0]-final_pos[i][6]/2, final_pos[i][1]-final_pos[i][7]/2
        rect = patches.Rectangle((x, y), final_pos[i][6], final_pos[i][7], linewidth=1, edgecolor=(float(final_pos[i][3]), float(final_pos[i][4]), float(final_pos[i][5])), facecolor='none')
        t = transforms.Affine2D().rotate_deg_around(final_pos[i][0], final_pos[i][1], final_pos[i][11]) + plt.gca().transData
        rect.set_transform(t)
        plt.gca().add_patch(rect)
    '''

    # Plot cube boundaries
    #rect = patches.Rectangle((96, 111), 3, 28, linewidth=1, edgecolor='gray', facecolor='none')
    #plt.gca().add_patch(rect)

    # Plot figure
    ax.set_xlim(80, 110)
    ax.set_ylim(140, 110)
    #if cfg.straight_traj:
    #    plt.xlim(85, 100)
    #    plt.ylim(130, 110)
    #else:
    #    plt.xlim(80, 105)
    #    plt.ylim(145, 105)
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    ax.legend()
    ax.grid()
    plot_name = 'results_traj.png'
    fig.savefig(plot_name, bbox_inches ='tight')
    print('training curve plots saved to: {}'.format(plot_name))


# ==================================== Main ===================================

if __name__ == '__main__':

    cfg = arg_parser().parse_args()
    print('Data folder:', cfg.output_dir)

    plot_training_curves(cfg.output_dir)

    # ============= Plot BOV =============

    run_idx = 0
    # params_per_iter has shape (num iter, num_cubes * num_params)
    params_per_iter = np.load(os.path.join(cfg.output_dir, 'run-{}'.format(run_idx), 'bestparams-log.npy'))
    print('params_per_iter shape', params_per_iter.shape)
    #bestparams_normalized = torch.from_numpy(params_per_iter[-2])
    bestparams_normalized = torch.from_numpy(params_per_iter)
    if cfg.cube:
        bestparams = denormalize(bestparams_normalized, get_nerf_max(), get_nerf_min())
    else:
        bestparams = bestparams_normalized
    plot_bov_traj(cfg, bestparams, bestparams_normalized)



