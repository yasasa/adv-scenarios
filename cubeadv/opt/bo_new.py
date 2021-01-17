import torch
import scipy
import numpy as np

from .optimizer import CubeOptimizer

from skopt import gp_minimize

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from cubeadv.utils import normalize

import time

from skopt import dump


nerf_max = np.array([100., 140.])
nerf_min = np.array([85., 125.])

class BONew(CubeOptimizer):
    def __init__(self,
                 dynamics,
                 sensor,
                 policy,
                 step_cost,
                 save_path,
                 car_init,
                 state_dim=3,
                 action_dim=1,
                 cube_dim=5,
                 integration_timesteps=200):
        super(BONew,
              self).__init__(dynamics, sensor, policy, step_cost, save_path,
                             state_dim, action_dim, cube_dim)
        self._T = integration_timesteps
        self._x0 = torch.from_numpy(np.array(car_init)).to(dtype=torch.float32).cuda()


    @torch.no_grad()    
    def _step(self, c, x0, grad=False):
        x = x0
        total_cost = 0
        
        if not torch.is_tensor(c):
            c = torch.Tensor(c).to(x.device)
        
        c = normalize(c.cpu(), nerf_max, nerf_min).to(x.device)
        

        for i in range(self._T):
            o = self._sensor(x, c) 
            u = self._policy(o)
            xn = self._dynamics(x, u)
            step_cost = self._step_cost.cost(xn, u)
            
            total_cost += step_cost
            x = xn

        return -total_cost


    def cost(self, c, x0):
        c = np.array(c, dtype=np.float32)
        c_string = np.array2string(c, precision=6, separator=',',
                      suppress_small=True)
        print(c_string)
        
        cost = self._step(c, x0, grad=False)
        return cost.item()
    
    
    @torch.no_grad()
    def f(self, c): # single step cost function
        return self.cost(c, self._x0)
    
    
    @torch.no_grad()
    def f_step(self, c): # single step cost function
        
        c = np.array(c, dtype=np.float32)
        c_string = np.array2string(c, precision=6, separator=',', suppress_small=True)
        print(c_string)
        
        x = self._x0
        if not torch.is_tensor(c):
            c = torch.Tensor(c).to(x.device)
            
        o = self._sensor(x, c) 
        u = self._policy(o)
        u = u.item()
        
        return - abs(u)
        



    def run(self, c0, x0, func, bounds, acq_func, iters, seed):
        
        cubes_init = c0
        
        c0 = torch.from_numpy(np.array(c0)).to(dtype=torch.float32).cuda()
        x0 = torch.from_numpy(np.array(x0)).to(dtype=torch.float32).cuda()
#         f = lambda c: self.cost(c, x0)

    
#         kernel = Matern(nu=2.5)
#         gpr = GaussianProcessRegressor(kernel=kernel, random_state=123)
    
        
        return gp_minimize(func,                     # the function to minimize
                          bounds,                # the bounds on each dimension of x
#                           base_estimator=gpr,
                          n_calls=iters,         # the number of evaluations of f
                          n_initial_points=5,    # the number of random initialization points
                          acq_func=acq_func,     # the acquisition function
#                           x0=cubes_init,
                          random_state=seed,      # the random seed
                          verbose=True)  
        
        


