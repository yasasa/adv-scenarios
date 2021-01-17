from pathlib import Path
import functools

import torch
import imageio
import matplotlib.pyplot as plt

import pytest

from cubeadv.sim.utils.path_map import PathMapCost
from cubeadv.utils.plotting import colorline

def img_comp(inp, target):
    mse = torch.nn.functional.mse_loss(inp, target)
    return mse
    
def plot_road(pm : PathMapCost, traj : torch.Tensor):
    fig, ax = plt.subplots()
    points = pm.points.squeeze()
    ax.plot(points[:, 0].numpy(), points[:, 1].numpy(), linestyle='--', color='black')
   # assert(False)
    lc = colorline(ax, traj[:, 0].numpy(), traj[:, 1].numpy(), cmap='viridis')
    
    plt.autoscale();
    xmin, xmax = ax.get_xlim()
    mid = (xmin + xmax) / 2
    xmin = min(xmin, mid - 0.1)
    xmax = max(xmax, mid + 0.1)
    ax.set_xlim(xmin, xmax)
    return fig, ax
    
def image_comparison_test(func):
    filename = f"{func.__name__}.png" 
    
    ref_path = Path("tests/ref_images", filename)
    save_path = Path("tests/actual_images", filename)
    
    save_path.parent.mkdir(exist_ok=True, parents=True)
    @functools.wraps(func)
    def comp_test(*args, **kwargs):
        ref = None
        if ref_path.exists():
            ref = torch.Tensor(imageio.v2.imread(ref_path))
            
        image = func(*args, **kwargs)
        
        if ref is None: 
            imageio.imwrite(save_path, image)
            assert("No reference images" and False)
        else:
            mse = img_comp(image, ref)
            imageio.imwrite(save_path, image)
            assert(f"Images differed by {mse}" and 
                        mse < pytest.image_comparision_thresh)
                     
    return comp_test
    
def make_file(dir, basename, suffix, ext):
    filename = f"{suffix}.{ext}"
    output_path = Path(dir, basename, filename)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    return output_path

def plotting_test(func):
    @functools.wraps(func)
    def test(*args, **kwargs):
        fig, axes, name_suffix = func(*args, **kwargs)
        if fig is not None:
            output_path = make_file("tests/actual_images", 
                                        func.__name__, name_suffix, "png")
            fig.savefig(output_path)
    return test
    
def driving_test(func):
    @functools.wraps(func)
    def test(*args, **kwargs):
        fig, axes, observations, name_suffix = func(*args, **kwargs)
        
        if fig is not None:
            output_path = make_file("tests/actual_images", 
                                         func.__name__, name_suffix, "png")
            fig.savefig(output_path)
            
        if observations is not None:
            output_path = make_file("tests/actual_images", 
                                         func.__name__, name_suffix, "gif")
            imageio.v2.mimwrite(output_path, observations)
    return test
    
    
def image_collection_test(func):
    @functools.wraps(func)
    def test(*args, **kwargs):
        fig, axes, observations, name_suffix = func(*args, **kwargs)
        
        if fig is not None:
            output_path = make_file("tests/actual_images", 
                                         func.__name__, name_suffix, "png")
            fig.savefig(output_path)
            
        if observations is not None:
            
            output_path_gif = make_file("tests/actual_images", 
                                             func.__name__, f"{name_suffix}", "gif")
            for i, im in enumerate(observations):
            
                output_path = make_file("tests/actual_images", 
                                             func.__name__, f"{name_suffix}_{i}", "png")
                imageio.v2.imwrite(output_path, im)
                
            
            imageio.v2.mimwrite(output_path_gif, observations)
    return test
       