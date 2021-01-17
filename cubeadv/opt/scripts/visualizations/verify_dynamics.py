#!/usr/bin/env python
from __future__ import print_function
import os
import sys
import argparse
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.append("../")
from generate_spawn_points import *

### Change below as necessary
args_expnum = 2
exp_dir = '../experiments/2020-1-10.%d/data' % args_expnum
num_exp = 1
start_exp = 5
end_exp = start_exp + num_exp
# test_dir = '../random_sampling/2019-11-30.0'
# iteration = 93
num_iter = 5
### Change above as necessary

globalPos = np.zeros((0, num_iter*100, 3))
localPos = np.zeros((0, num_iter*100, 3))
waypoints = np.zeros((0, num_iter*100, 3))
for i in range(start_exp, end_exp):
	posL =  np.zeros((0,3))
	posG = np.zeros((0,3))
	waypt = np.zeros((0,3))
	for j in range(num_iter):
		tmp = np.load('%s/ep_%d/GlobalState/%d.npy' % (exp_dir, i, j)).reshape(100,3)
		posL = np.concatenate((posL, tmp))
		tmp = np.load('%s/ep_%d/LocalState/%d.npy' % (exp_dir, i, j)).reshape(100,3)
		posG = np.concatenate((posG, tmp))
		tmp = np.load('%s/ep_%d/Waypoint/%d.npy' % (exp_dir, i, j)).reshape(100, 3)
		waypt = np.concatenate((waypt, tmp))
	globalPos = np.concatenate((globalPos, posL.reshape(1,num_iter*100,3)))
	localPos = np.concatenate((localPos, posG.reshape(1, num_iter*100, 3)))
	waypoints = np.concatenate((waypoints, waypt.reshape(1, num_iter*100, 3)))

list1 = [(158.7,127.3), (100.1,127.3), (99.3, 127.1), (98.7, 126.9), 
			 (97.4, 126.2), (95.9, 125), (95.1, 123.7), (94.5,122), 
			 (94.5,90.6), (98.4,90.6), (98.4, 107.1), (108.5,123.4), 
			 (158.7,123.4)]
list1_new = []
for i, j in list1:
	list1_new.append((j, i))
seg1 = gen_polygon(list1_new)

list2 = [(158.7, 139.4), (99.9, 139.5), (99.4, 139.7), (98.8, 140.2),
			 (98.3, 141.2), (98.3, 145), (94.5,145), (94.5, 140.7),
			 (95.2, 139.1), (96.3, 137.5), (97.4, 136.6), (98.3, 136.1),
		   	 (99.8, 135.6), (99.8, 135.6), (101.3, 135.6), (158.7, 135.5)]
list2_new = []
for i, j in list2:
	list2_new.append((j, i))
seg2 = gen_polygon(list2_new)

list3 = [(86.3, 145), (82.4, 145), (82.4, 90.6), (86.3, 90.6)]
list3_new = []
for i, j in list3:
	list3_new.append((j,i))
seg3 = gen_polygon(list3_new)

patch1 = patches.PathPatch(seg1, facecolor='orange', lw=2)
patch2 = patches.PathPatch(seg2, facecolor='orange', lw=2)
patch3 = patches.PathPatch(seg3, facecolor='orange', lw=2)


fig1 = plt.figure()
ax1 = fig1.add_subplot(121)
# ax1.imshow(map_img, extent=[60, 160, 80, 160])
ax1.add_patch(patch1)
ax1.add_patch(patch2)
ax1.add_patch(patch3)
ax1.plot(globalPos[0,:,1], globalPos[0,:,0], color='red')
ax1.plot(waypoints[0,:,1], waypoints[0,:,0], color='blue')
ax1.plot(localPos[0,:,1]+waypoints[0,:,1], localPos[0,:,0]+waypoints[0,:,0], color='green')
ax1.set_xlim(ax1.get_xlim()[::-1])
ax1.set_ylim(ax1.get_ylim()[::-1])
ax1.axis('equal')

ax11 = fig1.add_subplot(122)
ax11.plot(globalPos[0,:,2])

fig2 = plt.figure()
ax2 = fig2.add_subplot(131)
ax2.plot(localPos[0,:,0])

ax21 = fig2.add_subplot(132)
ax21.plot(localPos[0,:,1])

ax22 = fig2.add_subplot(133)
ax22.plot(localPos[0,:,2])

plot_step = 3
fig3 = plt.figure()
ax3 = fig3.add_subplot(211)
ax3.quiver(globalPos[0,::plot_step,1], globalPos[0,::plot_step,0], np.sin(np.deg2rad(globalPos[0,::plot_step,2])), np.cos(np.deg2rad(globalPos[0,::plot_step,2])), units='xy', scale=1, color='red')
ax3.quiver(localPos[0,::plot_step,1]+waypoints[0,::plot_step,1], localPos[0,::plot_step,0]+waypoints[0,::plot_step,0], np.sin(np.deg2rad(localPos[0,::plot_step,2]+waypoints[0,::plot_step,2])), np.cos(np.deg2rad(localPos[0,::plot_step,2]+waypoints[0,::plot_step,2])), units='xy', scale=1, color='green')
ax3.axis('equal')
ax31 = fig3.add_subplot(212)
ax31.quiver(localPos[0,::plot_step,1], localPos[0,::plot_step,0], np.sin(np.deg2rad(localPos[0,::plot_step,2])), np.cos(np.deg2rad(localPos[0,::plot_step,2])), units='xy', scale=1)
ax31.axis('equal')

plt.show()


## next ones to try:
# 2019-11-29.9/ep_44
# 2019-11-29.9/ep_94
# 2019-11-30.0/ep_5
# 2019-11-30.0/ep_75
# 2019-11-30.0/ep_77
# 2019-11-30.0/ep_93