#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

# upper_right and lower_left are (x, y)
# ------ x2, y2
# |           |
# x1, y1 ------
def gen_from_rect(lower_left, upper_right):
	x1, y1 = lower_left
	x2, y2 = upper_right
	assert(x2 > x1)
	assert(y2 > y1)
	x = np.linspace(x1, x2, int(x2-x1)+1)
	y = np.linspace(y1, y2, int(y2-y1)+1)
	xy = np.meshgrid(x, y)
	xy = np.transpose(np.array(xy), (1, 2, 0))
	return xy.reshape(-1, 2)

def det(a, b):
	return np.linalg.det(np.array([a, b]))

def calc_convex_hull(v, v0, v1, v2):
	a = 1.0 * (det(v, v2) - det(v0, v2)) / det(v1, v2)
	b = - 1.0 * (det(v, v1) - det(v0, v1)) / det(v1, v2)
	eps = 1e-5
	if a >= 0-eps and b >= 0-eps and a + b <= 1 + eps:
		return True
	else:
		return False

# corners: p0, p1, p2 and counter-clockwise
#        p2
#      /    \
#     /      p1
#    p0 -----

def gen_from_triangle(p0, p1, p2):
	min_x, min_y = min(p0[0], p1[0], p2[0]), min(p0[1], p1[1], p2[1])
	max_x, max_y = max(p0[0], p1[0], p2[0]), max(p0[1], p1[1], p2[1])
	over = gen_from_rect((min_x, min_y), (max_x, max_y))
	v0 = np.array(p0)
	v1 = np.subtract(p1, p0)
	v2 = np.subtract(p2, p0)
	ret = np.empty((0,2))
	for pt in over:
		if calc_convex_hull(pt, v0, v1, v2):
			ret = np.concatenate((ret, [pt]), axis=0)
	return ret

def get_building_spawn_points():
	triangles = [((101, 110), (106, 119), (101, 119)),
				 ((94, 122), (99, 122), (99, 127))]
	pts_tri = np.empty((0,2))
	for v0,v1,v2 in triangles:
		pts_tri = np.concatenate((pts_tri, gen_from_triangle(v0,v1,v2)), axis=0)
	tri, _ = np.unique(pts_tri, return_inverse=True, axis=0)

	rectangles = [((73, 88), (86, 145)),
				  ((86, 136), (190, 145)),
				  ((94, 88), (101, 119)),
				  ((99, 119), (190, 127)),
				  ((94, 119), (101, 122)),
				  ((99, 122), (101, 127))]
	pts_rect = np.empty((0,2))
	for lower, upper in rectangles:
		pts_rect = np.concatenate((pts_rect, gen_from_rect(lower, upper)), axis=0)
	rect, _ = np.unique(pts_rect, return_inverse=True, axis=0)

	all_pts = np.concatenate((tri, rect), axis=0)
	# set all z-coordinate to 2.0
	z = np.ones((all_pts.shape[0], 1)) * 2.0
	return np.hstack((all_pts, z))  # shape: 3000 x 3


def get_building_spawn_points_new():
	triangles = [((101, 110), (106, 119), (101, 119)),
				 ((94, 122), (99, 122), (99, 127))]
	pts_tri = np.empty((0,2))
	for v0,v1,v2 in triangles:
		pts_tri = np.concatenate((pts_tri, gen_from_triangle(v0,v1,v2)), axis=0)
	tri, _ = np.unique(pts_tri, return_inverse=True, axis=0)

	rectangles = [((96, 66), (100, 119)),
				  ((99, 119), (190, 127)),
				  ((96, 119), (100, 122)),
				  ((99, 122), (100, 127))]
				  # ((96, 64), (160, 69))]
	pts_rect = np.empty((0,2))
	for lower, upper in rectangles:
		pts_rect = np.concatenate((pts_rect, gen_from_rect(lower, upper)), axis=0)
	rect, _ = np.unique(pts_rect, return_inverse=True, axis=0)

	all_pts = np.concatenate((tri, rect), axis=0)
	# set all z-coordinate to 2.0
	z = np.ones((all_pts.shape[0], 1)) * 1.0
	return np.hstack((all_pts, z))  # shape: 3000 x 3

def plot_spawn_points():
	fig = plt.figure()
	ax0 = fig.add_subplot(111)
	ret = get_building_spawn_points_new()
	ax0.scatter(ret[:,0], ret[:,1])
	plt.show()

def get_spawn_points_from_file(filename):
	ret = []
	with open(filename, 'r') as file:
		for line in file:
			# strip away newline characters
			line = line.replace('\n', '')
			pt = [int(i) for i in line.split('\t')]
			# print(pt)
			assert(len(pt) == 6)
			ret.append(pt)
	return ret

# pts = get_building_spawn_points_new()
# for pt in pts:
# 	print("%d\t%d\t%d\t0\t0\t0"%(pt[0], pt[1], pt[2]))