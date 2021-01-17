import time
import carla
import copy

from queue import Queue
from copy import deepcopy
from threading import Lock, Condition

import numpy as np

from .raw_sensor import RawSensor
from .depth_lidar import FreeLidarSensor
from ..utils.data_processors import lidar_project2d, get_rot_mat, colour_lidar_quarter
from ..utils.car_params import CAR_SHAPE, CAR_VALUES


class FakeLidarSensor(RawSensor):
    """ Lidar sensor in free roam mode """

    def __init__(self,
                 world,
                 x_init):
        super(FakeLidarSensor, self).__init__(
            world, (100, 32), (3, ))

        self._world = world

        self._scan = FreeLidarSensor(self._world).read(x_init)

    def read(self, state) -> np.ndarray:
        return self._scan




