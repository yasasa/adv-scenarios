#!/usr/bin/env python
from __future__ import print_function
import os
import sys
import argparse
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append('../')
from data_processors import *

# plt.ion()

# For this particular test case, we have:
# current pos = Transform(Location(x=335.489990, y=273.743317, z=1.320625), Rotation(pitch=0.000000, yaw=90.000046, roll=0.000000))
# dest pos = Transform(Location(x=202.550003, y=55.840000, z=1.320625), Rotation(pitch=0.000000, yaw=179.999756, roll=0.000000))
# set:  Location(x=335.489990, y=273.743317, z=1.320625)
# NNPID: current location,  Location(x=335.489990, y=273.743317, z=2.692593)
# local planner: current location,  Location(x=77.000000, y=87.400002, z=2.000000)
# init way pt:  Waypoint(Transform(Location(x=88.384132, y=92.953827, z=0.000000), Rotation(pitch=0.000000, yaw=89.991280, roll=0.000000)))

# Camera 0 rotation is:  Rotation(pitch=40.958782, yaw=-179.534332, roll=0.481399)
# Camera 1 rotation is:  Rotation(pitch=-0.363537, yaw=-89.849915, roll=40.959785)
# Camera 2 rotation is:  Rotation(pitch=-40.958771, yaw=0.465697, roll=-0.481417)
# Camera 3 rotation is:  Rotation(pitch=0.363537, yaw=90.150085, roll=-40.959778)

def get_rot_mat(pitch, yaw, roll):
    ## degree to radian conversion
    yaw = np.deg2rad(yaw)
    pitch = np.deg2rad(pitch)
    roll = np.deg2rad(roll)
    ## z-axis
    yaw_mat = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])
    ## y-axis
    pitch_mat = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                          [0, 1, 0],
                          [-np.sin(pitch), 0, np.cos(pitch)]])
    ## x-axis
    roll_mat = np.array([[1, 0, 0],
                         [0, np.cos(roll), -np.sin(roll)],
                         [0, np.sin(roll), np.cos(roll)]])

    rot_mat = np.dot(np.dot(roll_mat, pitch_mat), yaw_mat)
    ## Normalization doesn't seem necessary. Comment out for now
    print("rot_mat before normalization: ", rot_mat)
    rot_mat[:,0] = rot_mat[:,0] / np.linalg.norm(rot_mat[:,0])
    rot_mat[:,1] = rot_mat[:,1] / np.linalg.norm(rot_mat[:,1])
    rot_mat[:,2] = rot_mat[:,2] / np.linalg.norm(rot_mat[:,2])
    return rot_mat

episode = 0
npy_num = 1
frame_num = 50

## test 0
# frame_cam1 = get_rot_mat(-0.363537, -90, 40.959785)

## test 50
# frame_cam1 = get_rot_mat(-0.017868, -90, 0.390352)

## test 99
# frame_cam1 = get_rot_mat(-3.000318, -90, 0.793992)

## test 50 (1.npy)
frame_cam1 = get_rot_mat(0.008060, -90, 0.050748)

## Load images: initial images for the four cameras
img0 = np.load('ep_%d/CameraRGB_0/%d.npy' % (episode, npy_num))[frame_num]
img1 = np.load('ep_%d/CameraRGB_1/%d.npy' % (episode, npy_num))[frame_num]
img2 = np.load('ep_%d/CameraRGB_2/%d.npy' % (episode, npy_num))[frame_num]
img3 = np.load('ep_%d/CameraRGB_3/%d.npy' % (episode, npy_num))[frame_num]
lidar = np.load('ep_%d/Lidar/%d.npy' % (episode, npy_num))[frame_num].reshape(-1,3)

## Figures for debugging
fig0 = plt.figure()
fig1 = plt.figure()
fig2 = plt.figure()
fig3 = plt.figure()
fig4 = plt.figure()
fig5 = plt.figure()
ax00 = fig0.add_subplot(141)
ax01 = fig0.add_subplot(142)
ax02 = fig0.add_subplot(143)
ax03 = fig0.add_subplot(144)
ax40 = fig4.add_subplot(141)
ax41 = fig4.add_subplot(142)
ax42 = fig4.add_subplot(143)
ax43 = fig4.add_subplot(144)
ax1 = fig1.add_subplot(111, projection='3d')
ax2 = fig2.add_subplot(111, projection='3d')
ax30 = fig3.add_subplot(141, projection='3d')
ax31 = fig3.add_subplot(142, projection='3d')
ax32 = fig3.add_subplot(143, projection='3d')
ax33 = fig3.add_subplot(144, projection='3d')
ax5 = fig5.add_subplot(111, projection='3d')

## Show images
ax00.imshow(np.transpose(img0, (1,0,2)))
ax01.imshow(np.transpose(img1, (1,0,2)))
ax02.imshow(np.transpose(img2, (1,0,2)))
ax03.imshow(np.transpose(img3, (1,0,2)))
ax40.imshow(np.transpose(img0, (1,0,2)))
ax41.imshow(np.transpose(img1, (1,0,2)))
ax42.imshow(np.transpose(img2, (1,0,2)))
ax43.imshow(np.transpose(img3, (1,0,2)))


global_r = 3
global_phi = 4
global_theta = 5

def appendSpherical_np(xyz):
    ptsnew = np.hstack((xyz, np.zeros((xyz.shape[0], 3))))
    for i in xyz[:,0]:
        # Check that the returned lidar range is less than 50m
        if i > 5000:
            print(i)
            assert(0)
    xy = np.square(xyz[:,0]) + np.square(xyz[:,1])
    ptsnew[:,global_r] = np.sqrt(xy + np.square(xyz[:,2]))
    # ptsnew[:,4] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    ptsnew[:,global_phi] = np.rad2deg(np.arctan2(xyz[:,2], np.sqrt(xy))) # for elevation angle defined from XY-plane up
    ptsnew[:,global_theta] = np.rad2deg(np.arctan2(xyz[:,1], xyz[:,0]))
    print("range of theta is: ", max(ptsnew[:,global_theta]), " and ", min(ptsnew[:,global_theta]))
    print("range of phi is: ", max(ptsnew[:,global_phi]), " and ", min(ptsnew[:,global_phi]))
    return ptsnew


IMG_X = 1280
IMG_Y = 720
CameraFOV = 90
# Consider 4x smaller: 1280 * 720 => 320 * 180 = 57600
#       or 5x smaller: 1280 * 720 => 256 * 144 = 36864

f = IMG_X /(2 * math.tan(CameraFOV * np.pi / 360))
Cu = IMG_X / 2
Cv = IMG_Y / 2
K = [[f, 0, Cu],
     [0, f, Cv],
     [0, 0, 1 ]]
K = np.array(K)

def colour_lidar_quarter (points, frame, index, display_array):
    points = np.dot(points, frame)
    points = np.array(np.unique(points, axis=0))

    coloured_cs = np.zeros_like(points)
    assert display_array.dtype == np.uint8  # the RGB images have 8-bit pixel values

    #get points
    pts_2d, ori_pts, cs = [], [], []

    pts_copy = points

    x, y, z = pts_copy[:,0], pts_copy[:,1], pts_copy[:,2]
    theta = np.rad2deg(np.arctan2(y,x))

    Rot = np.array([[-1., 0., 0.],
                    [0., 0., 1.],
                    [0., 1., 0.]])

    if index == 0: # front
        mask = np.logical_and(theta<45, theta>-45)
        Rot = np.dot(get_rot_mat(0, -90, 0), Rot)
    elif index == 1: # right
        mask = np.logical_and(theta>45, theta<135)
    elif index == 2: # back
        mask = np.logical_or(theta>135, theta<-135)
        Rot = np.dot(get_rot_mat(0, 90, 0), Rot)
    elif index == 3: # left
        mask = np.logical_and(theta<-45, theta>-135)
        Rot = np.dot(get_rot_mat(0, 180, 0), Rot)
    else:
        print("index not in range 0...3.")
        assert(0)

    pts_copy = np.dot(pts_copy[mask], Rot).T
    pts_copy = np.divide(pts_copy, pts_copy[2,:][None,:], out=np.zeros_like(pts_copy), where=pts_copy[2,:][None,:]!=0)
    pts_2d = np.dot(K, pts_copy).T

    img_mask = np.logical_and(np.logical_and(pts_2d[:,0]>0, np.logical_and(pts_2d[:,0]<display_array.shape[0], pts_2d[:,1]<display_array.shape[1])), pts_2d[:,1]>0)
    temp = pts_2d[img_mask]
    if index == 0:
        ax00.scatter(temp[:,0], temp[:,1])
    elif index == 1:
        ax01.scatter(temp[:,0], temp[:,1])
    elif index == 2:
        ax02.scatter(temp[:,0], temp[:,1])
    else:
        ax03.scatter(temp[:,0], temp[:,1])

    for pt in range(pts_2d.shape[0]):
        if pts_2d[pt,0] < 0 or pts_2d[pt,1] < 0:
            # cs.append(np.array([0, 0, 0]))
            cs.append(np.array([255, 255, 255]))
        elif int(pts_2d[pt,0]) < display_array.shape[0] and int(pts_2d[pt,1]) < display_array.shape[1]:
            cs.append(display_array[int(pts_2d[pt,0]), int(pts_2d[pt,1])])
            # print("point on image")
        else:
            # print("2D point (%d, %d) is outside of image." % (int(pts_2d[whatever,0]), int(pts_2d[whatever,1])))
            # cs.append(np.array([0, 0, 0]))
            cs.append(np.array([255, 255, 255]))

    cs = np.divide(cs, 255.0).reshape(-1,3)

    # Reapply filter for colours
    points = np.dot(points[mask], frame.T)
    return np.append(points, cs, axis=1)

# Colour point cloud
coloured_pts = colour_lidar_quarter(lidar, frame_cam1, 0, img0)
coloured_pts = np.concatenate((coloured_pts, colour_lidar_quarter(lidar, frame_cam1, 1, img1)))
coloured_pts = np.concatenate((coloured_pts, colour_lidar_quarter(lidar, frame_cam1, 2, img2)))
coloured_pts = np.concatenate((coloured_pts, colour_lidar_quarter(lidar, frame_cam1, 3, img3)))

ax5.scatter(coloured_pts[:,0], coloured_pts[:,1], coloured_pts[:,2], c=coloured_pts[:,3:6], marker='.')
ax5.set_xlabel('x')
ax5.set_ylabel('y')
ax5.set_zlabel('z')

pts_ori = lidar_project2d(coloured_pts)
print(pts_ori.shape)
ax1.scatter(pts_ori[:,0], pts_ori[:,1], pts_ori[:,2], c=pts_ori[:,3:6], marker='.')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')

plt.show()