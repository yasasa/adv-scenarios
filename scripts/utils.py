from ast import Module
import matplotlib.pyplot as plt
from policy_train_utils import plot_road_bg
import sys, os
import numpy as np

import torch
import imageio
from cubeadv.fields import NGPField
from cubeadv.sim.sensors import OrthographicCamera

from cubeadv.utils.plotting import colorline

from skimage.util import img_as_float32

def get_type_from_module(type : str, module : Module):
    objective = getattr(sys.modules[module.__name__], type)
    if objective is None:
        raise ValueError(f"Could not find {type} in {module}")

    return objective

def object_to_carla_loc(x):
    return x * 45. * 1.5 + torch.tensor([92.5, 132.5]).type_as(x)

def add_objects(object_locs, object_types, fig, ax, res_path="../resources"):
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from matplotlib.transforms import Affine2D, BboxTransform
    ims = []
    bboxes = []
    for obj_loc, obj_type in zip(object_locs, object_types):
        if "car" in obj_type:
            path = os.path.join(res_path, "car_top_down.png")
        else:
            path = os.path.join(res_path, "hydrant_top_down.png")
        im_ = plt.imread(path)
        im =  OffsetImage(im_, zoom=5./200.)
        ims.append(im)
        cloc = object_to_carla_loc(obj_loc[:2])
        print(obj_loc[3])

        tr = Affine2D().rotate(obj_loc[3])
        im.set_transform(tr)
        bbox = AnnotationBbox(im, cloc, frameon=False)
        bbox.set_transform(tr)
        bboxes.append(ax.add_artist(bbox))
    return bboxes

@torch.no_grad()
def overlay_objects(w, h, fields, transforms, bg=None, extent=30., center=(92.5, 132.5)):

    xc = center[0]
    yc = center[1]
    prescale = 25.

    camera = OrthographicCamera(w, h, extent*prescale, far_plane=10)

    if bg is None:
        rgb = torch.ones(h, w, 3).cuda()
    else:
        rgb = torch.tensor(bg).cuda()

    for field, tr in zip(fields, transforms):
        render_c2w  = torch.tensor([[0, -1., 0,  xc],
                                   [1, 0., 0.,  yc + 30.],
                                   [0., 0, 1.,   0.],
                                   [0., 0., 0., 0.]]).cuda()
        scale = torch.eye(3).type_as(tr)
        rmat2 = torch.tensor([[tr[3].cos(), 0., -tr[3].sin()],
                              [0., 1., 0.],
                              [tr[3].sin(), 0., tr[3].cos()]]).type_as(tr)
        render_c2w[:3, :3] = scale.matmul(rmat2.matmul(render_c2w[:3, :3]))

        tl = object_to_carla_loc(tr[:2])

        offset = torch.tensor([xc - tl[0], 0., tl[1] - yc]).float().cuda() * prescale
        offset = rmat2.matmul(offset)

        render_c2w[2, 3] = offset[2]
        render_c2w[0, 3] = xc + offset[0]

      # bs = field.render_batch_size
      # field.render_batch_size = 400

        rb = camera.read_internal(field, render_c2w)
        rgb[rb.hit.view(h, w)] = rb.rgb[0][rb.hit.view(h, w)]

      # field.render_batch_size = bs

    return rgb

def plot_road_bg(fig, ax, fields=None, transforms=None):
    LANE_WIDTH=4

    path_segments = np.array([[129.4, 132.5], [96.4, 132.5], [93.4, 130.7], [91.2, 128.5], [90.4, 125.5], [90.4, 101.]])

    path_vectors = path_segments[1:] - path_segments[:-1]
    segment_lengths = np.linalg.norm(path_vectors, axis=-1)
    path_vectors_normalized = path_vectors / segment_lengths.reshape(-1, 1)

    rot_path_vectors = path_vectors_normalized.dot(np.array([[0, -1],[1, 0]]))
    top_road_segments = np.copy(path_segments)
    top_road_segments[1:] -= rot_path_vectors * LANE_WIDTH/2
    top_road_segments[0] -= rot_path_vectors[0] * LANE_WIDTH / 2

    midpoint_vector = np.array([0, 132.5])
    bot_road_segments = (top_road_segments - midpoint_vector).dot(np.array([[1, 0],[0, -1]])) + midpoint_vector

    def segments_to_lines(seg):
        return seg[1:] - seg[:-1]

    top_road_lines = segments_to_lines(top_road_segments)
    bot_road_lines = segments_to_lines(bot_road_segments)
    side_road_segment = np.array([[90.4 - LANE_WIDTH/2, 101], [90.4 - LANE_WIDTH/2, 162 + LANE_WIDTH]])

    xmin=90.4 - 15.
    xmax=90.4 + 15.
    ymin=131.5 - 15.
    ymax=131.5 + 15.

    im = img_as_float32(imageio.imread("resources/carla_bg.png"))
    if fields is not None:
        im = overlay_objects(im.shape[1], im.shape[0], fields, transforms, bg=im, extent=27., center=(91.5, 132.5))

        im = im.detach().cpu()
    ax.imshow(im,extent=(xmin, xmax, ymax, ymin))

    # Plot data roads
    ax.plot(side_road_segment[:, 0], side_road_segment[:, 1], color='green', label="Road Edges", alpha=0.2, linestyle='--')
    ax.plot(top_road_segments[:, 0], top_road_segments[:, 1], color='green', alpha=0.2, linestyle='--')
    ax.plot(bot_road_segments[:, 0], bot_road_segments[:, 1], color='green', alpha=0.2, linestyle='--')
    plt.xlim(xmin, xmax)
    plt.ylim(ymax, ymin)


def get_bov_image(xs, path_map, fields=None, obj_transforms=None):
    plt.clf();
    plt.cla();

    fig, ax  = plt.subplots()
    plt.style.use('seaborn-whitegrid')
    plot_road_bg(fig, ax, fields=fields, transforms=obj_transforms)

    points = path_map.offset_map(2.)

    ax.plot(points[0, :, 0].cpu().numpy(), points[0, :, 1].cpu().numpy(), linestyle='--', label = "Center of the lane")
    
  #  ax.plot(xs[:, 0], xs[:, 1], color='orange', label="With Adv-Attack Trajectory")
    
    lc = colorline(ax, xs[:, 0].numpy(), xs[:, 1].numpy(), cmap='plasma')
    
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
  #  ax.set_xlim(80, 130)
  #  ax.set_ylim(160, 100)

   # ax.legend()
   # ax.grid()
    return fig, ax