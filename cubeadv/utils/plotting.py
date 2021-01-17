import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_road(fig, ax):
    LANE_WIDTH=5

    path_segments = np.array([[131.4, 129.5],  [98.4, 129.5], [95.4, 127.7], [93.2, 125.5], [92.4, 122.5], [92.4, 100.]])
    path_vectors = path_segments[1:] - path_segments[:-1]
    segment_lengths = np.linalg.norm(path_vectors, axis=-1)
    path_vectors_normalized = path_vectors / segment_lengths.reshape(-1, 1)

    rot_path_vectors = path_vectors_normalized.dot(np.array([[0, -1],[1, 0]]))
    top_road_segments = np.copy(path_segments)
    top_road_segments[1:] -= rot_path_vectors * LANE_WIDTH/2
    top_road_segments[0] -= rot_path_vectors[0] * LANE_WIDTH / 2
    midpoint_vector = np.array([0, 129.5 + LANE_WIDTH / 2])
    bot_road_segments = (top_road_segments - midpoint_vector).dot(np.array([[1, 0],[0, -1]])) + midpoint_vector

    def segments_to_lines(seg):
        return seg[1:] - seg[:-1]

    top_road_lines = segments_to_lines(top_road_segments)
    bot_road_lines = segments_to_lines(bot_road_segments)
    side_road_segment = np.array([[92.4 - LANE_WIDTH, 100], [92.4 - LANE_WIDTH, 159 + LANE_WIDTH]])

    path_segments = np.array([[120, 129], [115, 129]])

    ax.plot(side_road_segment[:, 0], side_road_segment[:, 1], color='black', label="Road Edges")
    ax.plot(top_road_segments[:, 0], top_road_segments[:, 1], color='black')
    ax.plot(bot_road_segments[:, 0], bot_road_segments[:, 1], color='black')
 
 
# Taken from:https://nbviewer.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    return segments   

def colorline(ax, x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''
    
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
           
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])
        
    z = np.asarray(z)
    
    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    
    ax.add_collection(lc);
    
    return lc
