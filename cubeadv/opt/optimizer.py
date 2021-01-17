import os
import logging
import pickle

import scipy.optimize


import torch
import numpy as np

try:
    import carla
    from bo.optimizers import BayesianOptimizer
    from ipopt import minimize_ipopt
except:
    pass

import traceback


class CubeOptimizer:
    def __init__(self, dynamics, sensor, policy, step_cost, save_path,
                 state_dim, action_dim, cube_dim, **optimizer_params):
        self._dynamics = dynamics
        self._sensor = sensor
        self._policy = policy
        self._step_cost = step_cost
        self._save_path = save_path
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._cube_dim = cube_dim
        self._base_logger_name = self.__class__.__name__

    def save(self, name, value):
        path = os.path.join(self._save_path, name)
        if os.path.isfile(path):
            current = np.load(path)
            new = np.concatenate([current, value], axis=0)
            np.save(path, new)
        else:
            np.save(path, value)

    def run(self, x0):
        raise NotImplementedError

    def log(self, logger_name=None):
        name = self._base_logger_name
        if logger_name:
            name = "{}.{}".format(self._base_logger_name, logger_name)
        return logging.getLogger(name)

    def save_traj(self, c0, x0, filename):
        raise NotImplementedError

    @classmethod
    def get(cls, key):
        for subclass in cls.__subclasses__():
            if subclass.__name__.lower() == key.lower():
                return subclass

        raise IndexError("No optimizer found: {}".format(key))
