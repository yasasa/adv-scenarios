from abc import ABC, abstractmethod
from typing import Union

import torch
from torch import Tensor

class BaseRadianceField(ABC):
    
    @abstractmethod
    def reload(self, cfg: Union[dict, str], *args, **kwargs):
        pass    
    
    def __call__(self, origins : Tensor, rays : Tensor, *args, **kwargs):
        return self.query(origins, rays, *args, **kwargs)
        
    @abstractmethod
    def query(self, origins : Tensor, rays : Tensor, *args, **kwargs):
        pass
    
    @abstractmethod
    def depth(self, origins : Tensor, rays : Tensor, *args, **kwargs):
        pass
    
    @abstractmethod
    def rgb(self, origins : Tensor, rays : Tensor, *args, **kwargs):
        pass
    
class MockRFNefObject(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_color = torch.nn.Parameter(torch.ones(3, 3).cuda(), requires_grad=True)
        
class MockRFPipelineObject(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nef = MockRFNefObject()
    
class MockRFObject(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pipeline = MockRFPipelineObject()
        
class MockRF(BaseRadianceField, torch.nn.Module):
    def __init__(self, num_obj):
        super().__init__()
        self.obj_fields = torch.nn.ModuleList([MockRFObject() for _ in range(num_obj)])
        self.B = 0.5*torch.ones(3, 3).cuda()
        self.D = 0.25*torch.randn(3, 3).cuda()
        
    def reload(self, cfg, *args, **kwargs):
        pass
    
    def set_transform_params(self, p):
        pass
    
    def depth(self, origins : Tensor, rays : Tensor, *args, **kwargs):
        depth = torch.matmul(origins, self.obj_fields[0].pipeline.nef.decoder_color) + torch.matmul(rays, self.D)
        
        return depth
        
    def rgb(self, origins : Tensor, rays : Tensor, *args, **kwargs):
        rgb = torch.matmul(origins,  self.obj_fields[0].pipeline.nef.decoder_color) + torch.matmul(rays, self.B)
        return rgb
    
    def query(self, origins : Tensor, rays : Tensor, *args, **kwargs):
        return self.rgb(origins, rays, *args, **kwargs), self.depth(origins, rays, *args, **kwargs)