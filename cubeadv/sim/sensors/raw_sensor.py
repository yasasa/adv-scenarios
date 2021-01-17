import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
import imageio

import abc


class RawSensor(nn.Module):
    """ Interface for a sensor object """

    def __init__(self, world, output_shape: Tuple[int, ...],
                 input_shape: Tuple[int, ...]):
        super().__init__()
        self._world = world
        self._output_shape = output_shape
        self._input_shape = input_shape

    @property
    def output_shape(self):
        return self._output_shape

    @property
    def input_shape(self):
        return self._input_shape

    def __call__(self, x, *args, **kwargs):
        return self.read(x, *args, **kwargs)

    def read(self, x, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError
        
    def forward(self, *args, **kwargs):
        return self.read(*args, **kwargs);

    def save_img(self, torch_img, filename):
        im_show = torch_img.cpu().detach().numpy()
        im_show = (im_show * 255).astype(np.uint8)
        imageio.imwrite(filename,im_show)
        
class Sensor(abc.ABC):
    
    @abc.abstractmethod
    def read(self, state):
        pass
    
    def __call__(self, *args, **kwargs):
        return self.read(*args, **kwargs)

    def save_img(self, torch_img, filename):
        im_show = torch_img.cpu().detach().numpy()
        im_show = (im_show * 255).astype(np.uint8)
        imageio.imwrite(filename,im_show)

