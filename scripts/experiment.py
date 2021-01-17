import os
import sys
import time
from copy import deepcopy

import torch
import numpy as np
import matplotlib as mpl
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from scripts.optimizers import RandomOptimizer, CEMOptimizer
import objectives

sys.path.append("..")
from configs.parser import arg_parser

def to8b(im):
    return (255 * np.clip(im, 0, 1)).astype(np.uint8)

class Experiment:
    def __init__(self, config, objective, init_texture_params=None, init_transform_params=None, profiler=None):
        self.config = config
        self.objective = objective
        self.prof = profiler
        
        assert(not (init_texture_params is None and init_transform_params is None))
        
        self.transform_params = None
        self.texture_params = None
        optim_params = []
        self.bounds = []
        if init_texture_params is not None:
            self.texture_params = init_texture_params.requires_grad_(True)
            optim_params.append({'params' : self.texture_params, 'lr': config.lr})
            min_c, max_c = objective.get_color_constraints()
            self.bounds.append({"min": min_c.to(objective.device).requires_grad_(False),
                                "max": max_c.to(objective.device).requires_grad_(False)})
        
        if init_transform_params is not None:
            self.transform_params = init_transform_params.requires_grad_(True)
            optim_params.append({'params':self.transform_params, 'lr':config.lr / 1000})
            min_tr, max_tr = objective.get_transform_constraints()
            self.bounds.append({"min": min_tr.to(objective.device).requires_grad_(False),
                                "max": max_tr.to(objective.device).requires_grad_(False)})
            
        

        # Optimizer
        if self.config.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(params=optim_params, lr=config.lr)
        elif self.config.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(params=optim_params, lr=config.lr, momentum=config.momentum)
        elif self.config.optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(params=optim_params, lr=config.lr, momentum=config.momentum)
        elif self.config.optimizer == 'random':
            self.optimizer = RandomOptimizer(optim_params, self.bounds)
        elif self.config.optimizer == 'cem':
            self.optimizer = CEMOptimizer(optim_params, self.bounds)
        else:
            raise NotImplementedError()

        if self.config.optimizer not in ['random', 'bo'] and config.scheduling > 0:
            self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, config.scheduling)
        else:
            self.sched = None

        # intermediary tracking
        self.param_trajectory = []
        if self.config.manual_log:
            self.obj_losses = []
            self.constr_losses = []
            self.debug_vars = []

        self.tb_writer = SummaryWriter(log_dir=self.config.output_dir)
        
    @property
    def params(self):
        if self.texture_params is None:
            print('a')
            return self.transform_params
        if self.transform_params is None:
            print('b')
            return self.texture_params
        return torch.cat([self.transform_params, self.texture_params])

    # ============================= Logging Utils =============================

    def __save_list(self, name, obj, itr=None):
        if itr is None:
            filename = os.path.join(self.config.output_dir, "{}-log".format(name))
        else:
            filename = os.path.join(self.config.output_dir, "{}-log-{}".format(name,itr))
        np.save(filename, obj, allow_pickle=True)

    def manual_log_data(self, itr, param_clone, obj_loss, constr_loss, save_debug_vars=False, is_best=False):
        # Update stat lists
        debug_vars = None
        if save_debug_vars:
            debug_vars = self.objective.get_debug_vars(itr, param_clone)
        self.debug_vars.append((itr, debug_vars))
        self.obj_losses.append(obj_loss)
        self.constr_losses.append(constr_loss)
        if self.config.save_param_traj:
            if type(param_clone) == list:
                param_clone = [param.cpu().numpy() for param in param_clone]
            else:
                param_clone = param_clone.cpu().numpy()
            self.param_trajectory.append(param_clone)

        # Save stats
        if (itr + 1) % self.config.save_params == 0 or itr == self.config.iterations - 1:
            print('Logging data at iteration', itr)
            losses_to_save = np.array(self.obj_losses)
            self.__save_list("obj_loss", losses_to_save)
            losses_to_save = np.array(self.constr_losses)
            self.__save_list("constr_loss", losses_to_save)

            if self.config.save_param_traj:
                param_traj_to_save = np.stack(self.param_trajectory) if len(self.param_trajectory) != 0 else None
                self.__save_list("params", param_traj_to_save, itr)
                print("SAVED optimization params over past iterations")

            if save_debug_vars:
                self.__save_list("debug", self.debug_vars)

        # Save best stats
        if is_best:
            if type(param_clone) == list:
                param_clone = [param.cpu().numpy() for param in param_clone]
            else:
                param_clone = param_clone.cpu().numpy()
            self.__save_list("bestparams", param_clone)

            if self.config.single_frame:
                if type(self.params) == list:
                    image = self.objective.render(param_clone)
                else:
                    image = self.objective.render(torch.from_numpy(param_clone).cuda())
                self.objective.sensor.save_img(image, 'tmp_rgb_out.png')
                print('Saved best attack render to tmp_rgb_out.png')

    def get_image_(self, p):
        if isinstance(p, np.ndarray):
            p = torch.from_numpy(p).to(self.objective.device)
        image = self.objective.render(p)
        image = to8b(image.cpu().detach().numpy())
        fig = mpl.figure.Figure()
        ax = fig.add_subplot()
        if not image is None:
            ax.imshow(image)
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])
        return fig

    def get_image(self, p):
        if isinstance(p, np.ndarray):
            p = torch.from_numpy(p).to(self.objective.device)
        image = self.objective.render(p)
        return image.permute(2, 0, 1)

    def get_plot(self, p):
        if isinstance(p, np.ndarray):
            p = torch.from_numpy(p).to(self.objective.device)
        fig, ax = self.objective.plot(p)
        return fig, ax

    def save_params(self, itr, params):
        if not isinstance(params, torch.Tensor):
            params = torch.cat([p.flatten() for p in params]).detach().cpu().numpy()
        else:
            params = params.detach().cpu().numpy()
        self.param_trajectory.append(params)
        self.__save_list("parameters", params, itr)

    def tb_log_data(self, itr, param_clone, loss, constr, save_debug_vars=False, is_best=False):
        self.tb_writer.add_scalar('obj_loss', loss, itr)
        self.tb_writer.add_scalar('constr_loss', constr, itr)
        if save_debug_vars:
            figure = self.get_image(param_clone)
            self.tb_writer.add_image('render', figure, itr)

            fig, ax = self.get_plot(param_clone)
            if fig is not None:
                self.tb_writer.add_figure('plot', fig, itr)


        if is_best:
            self.tb_writer.add_scalar('best_loss', loss, itr)
            self.save_params(0, param_clone)

        self.tb_writer.flush()

    def update_results(self, itr, obj_loss, constr_loss, param_clone, is_best=False):
        save_debug_vars = itr % self.config.debug_var_freq == 0

        if self.config.manual_log:
            self.manual_log_data(itr, param_clone, obj_loss, constr_loss, save_debug_vars, is_best)
        else:
            self.tb_log_data(itr, param_clone, obj_loss, constr_loss, save_debug_vars, is_best)

    # =========================== Constraint Losses ===========================

    def loss_constraints_log_barrier(self, _p):
        eps = 0.1
        mu = self.config.mu

        max_mask = (self.max_bounds - _p < 0.)
        min_mask = (_p - self.min_bounds < 0.)
        loss = - mu * torch.sum(torch.log(torch.masked_select(self.max_bounds - _p + eps, max_mask))) - mu * torch.sum(torch.log(torch.masked_select(_p - self.min_bounds + eps, min_mask)))

        return loss

    # Based on Eq 5.10 from https://www.cs.toronto.edu/~radford/ftp/ham-mcmc.pdf
    def loss_constraints_potential_energy(self, _p):

        r = torch.tensor(10)   # r is currently constant across all parameters

        maxconstr_mask = (self.max_bounds - _p < 0)
        maxconstr_violation_magnitudes = torch.masked_select(_p - self.max_bounds, maxconstr_mask)
        maxconstr_loss = torch.sum(torch.exp((r+1)*torch.log(r) + r*torch.log(maxconstr_violation_magnitudes)))

        minconstr_mask = (self.min_bounds - _p > 0)
        minconstr_violation_magnitudes = torch.masked_select(self.min_bounds - _p, minconstr_mask)
        minconstr_loss = torch.sum(torch.exp((r+1)*torch.log(r) + r*torch.log(minconstr_violation_magnitudes)))

        return minconstr_loss + maxconstr_loss


    def write_preopt(self, callback):
        l = -self.objective(self.params)
        constraint_violation = self.objective.constraint(self.params)

        self.update_results(0, l.item(), constraint_violation,
                                        self.params.clone().detach(), True)
        if callback:
            callback(0, l.item(), l.item(), constraint_violation.item(), self.params)

        if self.config.objective == 'PoseOptimization':
            preopt_img = self.objective.render_pre().permute(2, 0, 1)
            if preopt_img is not None:
                self.tb_writer.add_image("Reference pose", preopt_img)

    # ============================= Training Loop =============================

    def eval_constraints(self, p, ref):
        if self.config.constraint == 'barrier':
            constraint_loss = self.loss_constraints_log_barrier(p)
        elif self.config.constraint == 'potential':
            constraint_loss = self.loss_constraints_potential_energy(p)
        elif self.config.constraint == 'none':
            constraint_loss = torch.zeros(1, requires_grad=True).squeeze()
        else:
            raise NotImplementedError()

        return constraint_loss.type_as(ref)

    def maybe_clip_grad(self):
        if self.config.grad_clipping > 0.:
            if not isinstance(self.params, torch.Tensor):
                for param in self.params:
                    if type(param) is dict:
                        torch.nn.utils.clip_grad_norm_(param['params'], self.config.grad_clipping)
                    else:
                        torch.nn.utils.clip_grad_norm_(param, self.config.grad_clipping)
            else:
                torch.nn.utils.clip_grad_norm_(self.params, self.config.grad_clipping)

    def clone_params(self):
        param_clone = self.params.clone().detach()
        return param_clone

    def optimize_step(self, itr):
        opt = self.config.optimizer

        def eval(p):
            params = p
            objective_loss = -self.objective(params)
            return objective_loss

        if opt not in ['bo', 'cem']:
            self.optimizer.zero_grad()
            params = self.params
            objective_loss = eval(params)
            constraint_violation = self.objective.constraint(params)
            l = objective_loss
            if opt != 'random':
                l.backward()

            param_clone = self.clone_params()
            self.optimizer.step()

            if self.transform_params is not None:
                with torch.no_grad():
                    self.transform_params.data = self.objective.project_to_constraints(self.transform_params)
        else:
            param_clone, objective_loss = self.optimizer.step(eval)
            l = objective_loss

        if self.sched:
            self.sched.step()
        if self.prof:
            self.prof.step()

        return l.item(), objective_loss.item(), constraint_violation.item(), param_clone

    def run(self, callback=None):
        """
        Runs the optimization for N steps.
        """
        best_loss = np.inf
        eps_constr_loss = 10**(-6)

        self.write_preopt(callback)

        for k in range(1, self.config.iterations+1):
            loss, obj_loss, constr_loss, param_clone = self.optimize_step(k)

            is_best = False
            if loss < best_loss and constr_loss < eps_constr_loss:
                best_loss = loss
                is_best = True
                print('New best loss')

            self.update_results(k, obj_loss, constr_loss, param_clone, is_best)
            if callback is not None:
                callback(k, loss, obj_loss, constr_loss, self.params)
        self.tb_writer.close()

        return best_loss


def param_selector(cfg, objective):
    # Parameter setup
    transform_params = None
    texture_params = None
    if cfg.voxel_attack or cfg.objective in ['MultiFrameNGPColourAttack', 'SingleFrameNGPColourAttack']:
        texture_params = objective.params
    elif cfg.objective in ['MixedTransformAttack', 'RandomTransformAttack']:
        texture_params = objective.texture_params.clone().detach()
 #   else:
 #       print('Generating random initial params')
 #       texture_params = objective.get_random_params()
    
    if cfg.objective in ['TransformAndColorAttack', 'NGPTransformAttack', "RandomTransformAttack", "MixedTransformAttack"]:
        transform_params = objective.get_random_transform_params()
        
    return texture_params, transform_params


def mp_executor(run_idx, gpu_idx, cfg, objective_type, parent_output_dir, callback=None, multi_run=False):
    if multi_run:
        cfg.output_dir = os.path.join(parent_output_dir, 'run-{}'.format(run_idx))
    else:
        cfg.output_dir = parent_output_dir
    # create logging dir
    os.makedirs(cfg.output_dir, exist_ok=True)

    cfg = deepcopy(cfg)
    torch.manual_seed(cfg.seed + run_idx)
    np.random.seed(cfg.seed + run_idx)
    
    objective = objective_type(cfg)
    objective.init()
    if gpu_idx != 0:
        objective.set_device(f"cuda:{gpu_idx}")

    if cfg.profile:
        prof = torch.profiler.profile(schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2), on_trace_ready=torch.profiler.tensorboard_trace_handler(f'profiling/prof-log-{run_idx}'), record_shapes=True, with_stack=True, profile_memory=True)
        prof.start()
    else:
        prof = None

    init_texture_params, init_transform_params = param_selector(cfg, objective)
    exp = Experiment(cfg, objective, init_texture_params, init_transform_params, profiler=prof)
    best_loss_curr_run = exp.run(callback)

    if prof:
        prof.stop()

    del exp
    del objective
    del init_texture_params
    del init_transform_params

    return best_loss_curr_run


def get_objective_type(cfg):
    objective = getattr(sys.modules[objectives.__name__], cfg.objective)
    if objective is None:
        raise ValueError(f"Invalid objective function {cfg.objective}")

    return objective
    
if __name__ == '__main__':
    #torch.autograd.set_detect_anomaly(True)
    parser = arg_parser()
    cfg = parser.parse_args()
    print(cfg)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    objective_type = get_objective_type(cfg)

    print("Objective type:", objective_type)

    def callback(itr, loss, obj_loss, constr_loss, param):
        print('Iter: {}, Loss: {:.5f}, Obj Loss: {:.5f}, Constr Loss: {:.5f}'.format(itr, loss, obj_loss, constr_loss))
        print ('---')

    if cfg.exp_name is None:
        _, filename = os.path.split(cfg.cfg)
        file = os.path.splitext(filename)[0]
        cfg.exp_name = file

    parent_output_dir = os.path.join(cfg.output_dir, cfg.exp_name, cfg.objective, f"seed-{cfg.seed}")

    if cfg.init_params is not None:
        objective = objective_type(cfg)
        exp = Experiment(cfg, objective, torch.load(cfg.init_params))
        start = time.time()
        best_loss = exp.run(callback)
        print('Best loss from current run: {:.4f}, runtime: {} min'.format(best_loss, round((time.time()-start)/60,2)))

    elif cfg.process_count != 0:
        try:
            mp.set_start_method('spawn', force=True)
        except:
            pass
        if cfg.process_count < 0:
            cfg.process_count = torch.cuda.device_count()

        with mp.Pool(processes=cfg.process_count-1, maxtasksperchild=1) as pool:
            args = []
            for i in range(cfg.num_rand_inits):
                gpu_idx = (i % (torch.cuda.device_count() - 1)) + 1
                args.append((i, gpu_idx, cfg, objective_type))
            losses = pool.starmap(mp_executor, args)
            print(losses)

    else:
        best_loss = np.inf
        for i in range(cfg.num_rand_inits):
            print('======================= Random Start {} ======================='.format(i))
            start = time.time()
            best_loss_curr_run = mp_executor(i, 0, cfg,  objective_type, parent_output_dir, callback, cfg.num_rand_inits > 1)
            best_loss = min(best_loss_curr_run, best_loss)
            print('Best loss from current run: {:.4f}, runtime: {} min'.format(best_loss_curr_run, round((time.time()-start)/60,2)))

        print('Best loss from all runs: {:.4f}'.format(best_loss))



