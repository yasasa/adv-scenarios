import torch
import scipy
import numpy as np

from .optimizer import CubeOptimizer

from skopt import gp_minimize
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor

import time

class BO(CubeOptimizer):
    def __init__(self,
                 dynamics,
                 sensor,
                 policy,
                 step_cost,
                 save_path,
                 state_dim=3,
                 action_dim=1,
                 cube_dim=4,
                 integration_timesteps=200):
        super(BO,
              self).__init__(dynamics, sensor, policy, step_cost, save_path,
                             state_dim, action_dim, cube_dim)
        self._T = integration_timesteps


    @torch.no_grad()    
    def _step(self, c, x0, grad=False):
        x = x0
        total_cost = 0
        
        if not torch.is_tensor(c):
            c = torch.Tensor(c).to(x.device)
           

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


    def run(self, c0, x0):
        c0 = torch.from_numpy(c0).to(dtype=torch.float32).cuda()
        x0 = torch.from_numpy(x0).to(dtype=torch.float32).cuda()
        f = lambda c: self.cost(c, x0)


        right_bounds = [[(96.0, 99.0), (111.0, 129.0), (0.5, 2.0), (0, 1.0), (0, 1.0), (0, 1.0), (1.0, 2.0), (1.0, 2.0), (1.0, 2.0), 
                         (-30.0, 30.0), (-30.0, 30.0), (-30.0, 30.0)] for _ in range(15)]
#         left_bounds = [[(87.0, 89.0), (113.0, 125.0), (0.5, 2.0), (1.0, 2.0), (1.0, 2.0), (1.0, 2.0), (0, 1.0), (0, 1.0), (0, 1.0)] for _ in range(6)]
#         bounds = right_bounds + left_bounds
        bounds = right_bounds
        bounds = sum(bounds, [])
    
#         kernel = Matern(nu=2.5)
#         gpr = GaussianProcessRegressor(kernel=kernel, random_state=123)
    
        
        opt = gp_minimize(f,                     # the function to minimize
                          bounds,      # the bounds on each dimension of x
#                           base_estimator=gpr,
                          acq_func="EI",         # the acquisition function
                          n_calls=200,           # the number of evaluations of f
                          n_random_starts=200,   # the number of random initialization points
#                           noise=0.1**2,        # the noise level (optional)
                          random_state=123,      # the random seed
                          verbose=True)  
        


