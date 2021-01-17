import pytest

import torch
import numpy as np

from test_utils import plotting_test

from cubeadv.sim.utils import PathMapCost
import matplotlib.pyplot as plt
from matplotlib import colors

data = [
    (torch.tensor([[[0., 0.], [1., 0.]]]), torch.tensor([[0.5, 0.5]]), torch.tensor([[0.5, 0]]), torch.tensor([[-1.]])),
    (torch.tensor([[[0.,  0.], [0.5, 0.], [1., 0.5], [1., 1.]]]), 
        torch.tensor([[1.,  0.], [0.8, 0.7], [1., 1.]]),
        torch.tensor([[0.75, 0.25], [1., 0.7], [1., 1.]]),
        torch.tensor([[1.], [-1.], [1.]]))
]

@pytest.mark.parametrize("map,x,projection,sign", data)
@plotting_test
def test_path_map(map, x, projection, sign, request):
    pm = PathMapCost(map)
    
    dist, latdist, id = pm.projection(x)
    v = projection - x
    latdist_expected = sign.squeeze() * v.norm(dim=-1)
    
    v2 = projection[None] - map.squeeze()[:-1, None, :] 
    dist_expected, id_expected = v2.norm(dim=-1).min(dim=0)
    
    assert(torch.allclose(latdist, latdist_expected, atol=1e-4))
    assert(torch.allclose(dist, dist_expected, atol=1e-4))
    
    # Create grid of points
    xs = torch.linspace(-1, 2, 100)
    xm, ym = torch.meshgrid(xs, xs, indexing='xy')
    xy = torch.stack([xm, ym], dim=-1)
    xy = xy.view(-1, 2)
    dist, latdist, id = pm.projection(xy)
    latdist = latdist.view(100, 100)
    
    fig, ax = plt.subplots()
    cmap = ax.imshow(latdist, origin='lower', norm=colors.CenteredNorm() ,extent=[-1, 2, -1, 2], cmap='seismic')
    mnp = map.squeeze().numpy()
    ax.plot(mnp[:, 0], mnp[:, 1])
    
    fig.colorbar(cmap)
    return fig, ax, request.node.callspec.id
   