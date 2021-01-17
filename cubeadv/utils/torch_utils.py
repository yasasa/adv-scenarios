import torch

from dataclasses import dataclass
from typing import List, Tuple

# Utilities to use pytorch functional api with neural networks:
# https://discuss.pytorch.org/t/hvp-w-r-t-model-parameters/83520/4


@dataclass
class FuncParamStore:
    param_vector: torch.Tensor
    param_counts: Tuple[int]
    param_names: Tuple[int]
    param_shapes: Tuple[torch.Size]

def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)

def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])

def make_functional(model, param_filter=None, verbose=False) -> FuncParamStore:
    """
    Allow the parameters of a model to be modified at runtime
    Args:
        model: Model to modify.
        param_filter: Function that accepts the name of a parameter, will
                      return true if the parameter is modifiable.
                      Default: Includes every parameter.
    Returns:
        Tuple of (List of current parameters, List of metadata), each metadata
        is a tuple of (parameter name, parameter shape, parameter number of
        elements).
    """
    if param_filter is None:
        param_filter = lambda _: True
        
    orig_params = []
    param_counts, param_shapes, param_names = [], [], []

    for name, p in model.named_parameters():
        if param_filter(name):
            orig_params.append(p.data)
            param_counts.append(p.data.numel())
            param_shapes.append(p.data.shape)
            param_names.append(name)
    
    param_vector = torch.cat([p.flatten() for p in orig_params])

    for name in param_names:
        del_attr(model, name.split("."))
    
    params = FuncParamStore(param_vector, param_counts, param_names, param_shapes)

    # For debugging
    model._cubeadv_functional = True
    model._cubeadv_metadata = params

    return params

def set_weights(model, params: FuncParamStore, p : torch.Tensor):
    """
    Sets the weights of a model that was made functional prior.
    Args:
        model: Model to set the weights of, make_functional must have been called.
        metadata: List of meta data returned by make_functional.
        p: Vector of parameters, the total length is the sum of the number of
           elements of each param in metadata.
    """
    ci = 0
    for name, shape, numel in zip(params.param_names, params.param_shapes, params.param_counts):
        set_attr(model, name.split("."), p[ci: ci+numel].view(shape))
        ci += numel