#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import cv2
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

import random
from itertools import groupby

import os
##
import math
import numpy.ma as ma

### For adding colours to lidar point cloud
NUM_IMG = 4
IMG_X = 1280 #1280
IMG_Y = 720 # 720
CameraFOV = 90
# 1280 * 720 => 320 * 180
f = IMG_X /(2 * math.tan(CameraFOV * np.pi / 360))
Cu = IMG_X / 2
Cv = IMG_Y / 2
K = [[f, 0, Cu],
     [0, f, Cv],
     [0, 0, 1 ]]
K = np.array(K)

### For running CNN on lidar point cloud
HAngle_offset = 3.6 #0.36#3.6 # 360 / 100
VAngle_offset = 40 / 31.
Upper_FOV = 30
Lower_FOV = -10

### Change this boolean for training vs validation
is_validate = True
##################################################
global_r = 3 if is_validate == False else 6
global_phi = 4 if is_validate == False else 7
global_theta = 5 if is_validate == False else 8

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
    # print("rot_mat before normalization: ", rot_mat)
    # rot_mat[:,0] = rot_mat[:,0] / np.linalg.norm(rot_mat[:,0])
    # rot_mat[:,1] = rot_mat[:,1] / np.linalg.norm(rot_mat[:,1])
    # rot_mat[:,2] = rot_mat[:,2] / np.linalg.norm(rot_mat[:,2])
    return rot_mat


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
    # print("range of theta is: ", max(ptsnew[:,global_theta]), " and ", min(ptsnew[:,global_theta]))
    # print("range of phi is: ", max(ptsnew[:,global_phi]), " and ", min(ptsnew[:,global_phi]))
    return ptsnew


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


# This is only an approximation (cleaned-up version of the xyz locations)
def spherical2cartesian(rarray, theta, phi):
    x = rarray * np.sin(np.deg2rad(90-phi)) * np.cos(np.deg2rad(theta))
    y = rarray * np.sin(np.deg2rad(90-phi)) * np.sin(np.deg2rad(theta))
    z = rarray * np.cos(np.deg2rad(90-phi))
    return np.array([x,y,z]).T

# in training mode: points: (x, y, z, R, G, B, D, phi, theta)
# in validation mode: points: (x, y, z, R, G, B, D)
# theta: horizontal angle
def augment_pts(points, theta):
    vrange = np.linspace(Lower_FOV, Upper_FOV, 32)    # 32 x 1
    ret_depth = np.ones(32) * 60.   # 32 x 1
    ret_rgb = np.ones((32, 3))     # 32 x 3
    ret_xyz = spherical2cartesian(ret_depth, theta, vrange) # 32 x 3

    # this channel found no points within the 50m range
    if points.shape[0] == 0:
        # pseudo xyz coordinates, rgb=(0, 0, 0), depth=60, phi=vrange, theta=theta
        return np.hstack((ret_xyz, ret_rgb, ret_depth.reshape(32,1), vrange.reshape(32,1), np.ones(32).reshape(32,1) * theta))  # 32 x 9

    # sort by elevation angle
    temp_points = np.array(points[points[:,global_phi].argsort()])
    # print(temp_points[:,global_phi])
    idx = 0
    for i in range(32):
        if idx < temp_points.shape[0] and abs(temp_points[idx,global_phi]-vrange[i]) < VAngle_offset/2.:
            ret_depth[i] = temp_points[idx,global_r]
            ret_xyz[i] = temp_points[idx,0:3]
            ret_rgb[i] = temp_points[idx,3:6]
            idx += 1

    if idx != temp_points.shape[0]:
        pass
        #print("max: ", max(temp_points[:,global_theta]))
        #print("min: ", min(temp_points[:,global_theta]))
        #print("actual num points: ", temp_points.shape[0])
        #print("recorded num points: ", idx)
        #print("missing points!")
    if max(temp_points[:,global_phi]) - min(temp_points[:,global_phi]) > 40.5:
        pass
        #print("max: ", max(temp_points[:,global_phi]))
        #print("min: ", min(temp_points[:,global_phi]))
        i#print("range too big!")
    return np.hstack((ret_xyz, ret_rgb, ret_depth.reshape(32,1), vrange.reshape(32,1), np.ones(32).reshape(32,1) * theta))  # 32 x 9

def lidar_project2d(points):
    mask_0 = np.logical_not(np.logical_and(np.logical_and(points[:,0]==0, points[:,1]==0), points[:,2]==0))
    points = points[mask_0] # 32 channel * 100 pts/channel

    lidar_data = np.array(points)   # n x 6
    spherical_pts = appendSpherical_np(lidar_data) # n x 9

    # sort by angle in XY-plane
    spherical_pts = spherical_pts[spherical_pts[:,global_theta].argsort()]

    pts_to_keep = spherical_pts
    mask_0 = np.logical_not(np.logical_and(np.logical_and(pts_to_keep[:,0]==0, pts_to_keep[:,1]==0), pts_to_keep[:,2]==0))
    pts_to_keep = pts_to_keep[mask_0]

    ret = np.zeros((0, 9))

    ## Clean up pts_to_keep by HAngle_offset
    hrange = np.arange(180, -179, -HAngle_offset)
    for hangle in hrange:
        # Get all the points belonging to the same column (same theta)
        m = np.abs(pts_to_keep[:,global_theta] - hangle) < HAngle_offset/2.
        channel_pts = pts_to_keep[m]  # M x 9
        channel_ret = augment_pts(channel_pts, hangle)  # 32 x 9
        ret = np.vstack((ret, channel_ret))
    return ret     # 3200 x 9


# num_batch per episode
def load_data_cnn_lidar (path, episode_start, episode_end, batch_num_start, batch_num_end):
    ## Training Data
    data_ctrl = []
    data_pts = []

    # episodes
    for ep in range(episode_start, episode_end, 1):
        # images - 100 per batch
        for i in range(batch_num_start, batch_num_end, 1):   # 2 to 43
            ## test if the file exist
            filePath = path + str(ep) + '/Coloured_Spherical/' + str(i) + '.npy'
            if not os.path.exists(filePath):
                print(filePath,'doesnt exist')
                print('its ok, skip')
                continue
            print(filePath)
            pts = np.load(path + str(ep) + '/Coloured_Spherical/' + str(i) + '.npy')
            # print("pts shape is: ", pts.shape)
            ctrls = np.load(path + str(ep) + '/Control/' + str(i) + '.npy')
            # print("ctrls shape is: ", ctrls.shape)

            # control
            for c in ctrls:
                data_ctrl.append(c)

            # point cloud
            count = 0
            for pts_perBatch in pts:
                data_pts.append(pts_perBatch)

    ## control: [throttle, steer, brake, speed]
    ctrl = np.asarray(data_ctrl, dtype=np.float32)
    ctrl = ctrl.reshape(ctrl.shape[0],  -1)
    labels = ctrl[:, 1]  # steering                              # (N, 1)

    ## 3d point cloud
    points = np.asarray(data_pts, dtype=np.float32)                 # (N, 32, 100, 4)
    # print('points shape: ', points.shape)

    ## check if labels are legit
    # solve the problem of having nan stored in steering right after respawn to new position
    mask2 = np.logical_not(np.isnan(labels))
    labels = labels[mask2]
    points = points[mask2]
    # print('ctrl shape after mask nan: ', labels.shape)
    # print('points shape after mask nan: ', points.shape)

    for c in labels:
        if not (c>=-1 and c<=1):
            print(c)

    data_labels = torch.from_numpy(labels)  # steering
    print(points.shape)
    points = np.transpose(points, (0, 3, 1, 2))
    # points = points.transpose(points, (0, 2, 3, 1))
    print(points.shape)
    print("here")
    data_points = torch.from_numpy(points)  # Lidar detection 3d points
    print("load data lidar, points shape is: ", data_points.shape)

    return data_points, data_labels


# def colour_lidar (points, images):
#     points = np.array(np.unique(points, axis=0))

#     ############################################################################
#     x, y, z = points[:,0], points[:,1], points[:,2]

#     theta = np.rad2deg(np.arctan2(y,x))

#     masks = [np.logical_and(np.logical_and(theta>=-135, theta<-45), y<0),
#              np.logical_and(np.logical_and(theta>=-45, theta<45), x>0),
#              np.logical_and(np.logical_and(theta>=45, theta<135), y>0),
#              np.logical_and(np.logical_or(theta<-135, theta>=135), x<0)]

#     coloured_cs = np.zeros_like(points)

#     for img_idx in range(NUM_IMG):
#         rgbs = images[img_idx]
#         # print('Shapes: RGB: ', rgbs.shape)      # 180 * 320
#         # print('Data types: RGB: ', rgbs.dtype)  # uint8
#         assert rgbs.dtype == np.uint8  # the RGB images have 8-bit pixel values
#         # display_array = cv2.cvtColor(rgbs, cv2.COLOR_BGR2RGB)  # convert images from BGR to RGB using opencv
#         display_array = rgbs
#         # print('image shape = ', display_array.shape)

#         #get points
#         pts_2d = []
#         ori_pts = []
#         cs = []
#         mask = masks[img_idx]

#         Rot = np.zeros(shape=(3, 3))
#         # # Rotate the lidar xyz frame to image xyz frame
#         # # Front
#         # if img_idx == 0:
#         #     Rot = np.array([[1., 0., 0.],
#         #                     [0., 0., -1.],
#         #                     [0., 1., 0.]])
#         # # Right
#         # if img_idx == 1:
#         #     Rot = np.array([[0., 1., 0.],
#         #                     [0., 0., -1.],
#         #                     [-1., 0., 0.]])
#         # # Back
#         # if img_idx == 2:
#         #     Rot = np.array([[-1., 0., 0.],
#         #                     [0., 0., -1.],
#         #                     [0., -1., 0.]])
#         # # Left
#         # if img_idx == 3:
#         #     Rot = np.array([[0., -1., 0.],
#         #                     [0., 0., -1.],
#         #                     [1., 0., 0.]])

#         # Debug: 09-24
#         # Rotate the lidar xyz frame to image xyz frame
#         # Front
#         if img_idx == 0:
#             Rot = np.array([[-1., 0., 0.],
#                             [0., 0., -1.],
#                             [0., 1., 0.]])
#         # Right
#         if img_idx == 1:
#             Rot = np.array([[0., -1., 0.],
#                             [0., 0., -1.],
#                             [-1., 0., 0.]])
#         # Back
#         if img_idx == 2:
#             Rot = np.array([[1., 0., 0.],
#                             [0., 0., -1.],
#                             [0., -1., 0.]])
#         # Left
#         if img_idx == 3:
#             Rot = np.array([[0., 1., 0.],
#                             [0., 0., -1.],
#                             [1., 0., 0.]])


#         pt2 = np.dot(Rot, np.multiply(points, np.vstack((mask, mask, mask)).T).T) # 3 x 1900
#         pt2 = np.divide(pt2, pt2[2,:][None,:], out=np.zeros_like(pt2), where=pt2[2,:][None,:]!=0)
#         pts_2d = np.matmul(K, pt2).T

#         # print(pts_2d.shape)
#         for whatever in range(pts_2d.shape[0]):
#             if pts_2d[whatever,0] < 0 or pts_2d[whatever,1] < 0:
#                 cs.append(np.array([0, 0, 0]))
#                 continue
#             if int(pts_2d[whatever,0]) < display_array.shape[0] and int(pts_2d[whatever,1]) < display_array.shape[1]:
#                 cs.append(display_array[int(pts_2d[whatever,0]), int(pts_2d[whatever,1])])
#             else:
#                 # print("2D point (%d, %d) is outside of image." % (int(pts_2d[whatever,0]), int(pts_2d[whatever,1])))
#                 cs.append(np.array([0, 0, 0]))

#         cs = np.divide(cs, 255.0)

#         # Reapply filter for colours
#         cs = np.multiply(cs, np.vstack((mask, mask, mask)).T) # 3 x n
#         coloured_cs = coloured_cs + cs.reshape(-1, 3)

#     return np.append(points, coloured_cs, axis=1)  # n x 6








# def augment_pts_2(points, phi):
#     hrange = np.linspace(-180, 180, 1001)
#     ret_depth = np.ones(1000) * 60.   # 32 x 1
#     ret_rgb = np.ones((1000, 3))     # 32 x 3
#     ret_xyz = spherical2cartesian(ret_depth, hrange, phi) # 32 x 3

#     # this channel found no points within the 50m range
#     if points.shape[0] == 0:
#         # pseudo xyz coordinates, rgb=(0, 0, 0), depth=60, phi=vrange, theta=theta
#         return np.hstack((ret_xyz, ret_rgb, ret_depth.reshape(1000,1), np.ones(1000).reshape(1000,1) * phi, hrange.reshape(1000,1)))  # 32 x 9

#     # sort by elevation angle
#     temp_points = np.array(points[points[:,global_theta].argsort()])
#     # print(temp_points[:,global_phi])
#     idx = 0
#     for i in range(1001):
#         # print("idx: ", idx)
#         # print("vrange i: ", i, ": ", vrange[i])
#         # if (idx < temp_points.shape[0]):
#             # print(temp_points[idx,global_phi])
#         if idx < temp_points.shape[0] and abs(temp_points[idx,global_theta]-hrange[i]) < HAngle_offset/2.:
#             ret_depth[i] = temp_points[idx,global_r]
#             ret_xyz[i] = temp_points[idx,0:3]
#             ret_rgb[i] = temp_points[idx,3:6]
#             idx += 1
#         if idx < temp_points.shape[0] and abs(temp_points[idx,global_theta]-hrange[i]) >= HAngle_offset/2.:
#             print(temp_points[idx,global_theta])

#     if idx != temp_points.shape[0]:
#         print("max: ", max(temp_points[:,global_theta]))
#         print("min: ", min(temp_points[:,global_theta]))
#         print("actual num points: ", temp_points.shape[0])
#         print("recorded num points: ", idx)
#         print("missing points!")
#     # if max(temp_points[:,global_phi]) - min(temp_points[:,global_phi]) > 40.5:
#     #     print("max: ", max(temp_points[:,global_theta]))
#     #     print("min: ", min(temp_points[:,global_theta]))
#     #     print("range too big!")
#     return np.hstack((ret_xyz, ret_rgb, ret_depth.reshape(1000,1), np.ones(1000).reshape(1000,1) * phi, hrange.reshape(1000,1)))  # 32 x 9

# def lidar_project2d_2(points):
#     mask_0 = np.logical_not(np.logical_and(np.logical_and(points[:,0]==0, points[:,1]==0), points[:,2]==0))
#     points = points[mask_0] # 32 channel * 100 pts/channel

#     lidar_data = np.array(points)   # n x 6
#     spherical_pts = appendSpherical_np(lidar_data) # n x 9

#     # sort by angle in XY-plane
#     pts_to_keep = spherical_pts[spherical_pts[:,global_phi].argsort()]

#     mask_0 = np.logical_not(np.logical_and(np.logical_and(pts_to_keep[:,0]==0, pts_to_keep[:,1]==0), pts_to_keep[:,2]==0))
#     pts_to_keep = pts_to_keep[mask_0]

#     ret = np.zeros((0, 9))

#     ## Clean up pts_to_keep by VAngle_offset
#     vrange = np.linspace(Lower_FOV, Upper_FOV, 32)    # 32 x 1
#     for vangle in vrange:
#         # print("hangle: ", hangle)
#         # Get all the points belonging to the same column (same theta)
#         m = np.abs(pts_to_keep[:,global_phi] - vangle) < VAngle_offset/2.
#         channel_pts = pts_to_keep[m]  # M x 9
#         # print("channel_pts: ", channel_pts.shape)
#         channel_ret = augment_pts_2(channel_pts, vangle)  # 32 x 9
#         # if abs(hangle - 180) > 0.5:
#         ret = np.vstack((ret, channel_ret))
#         # else:
#         #     for a in range(channel_ret.shape[0]):
#         #         if channel_ret[a,6] != 60:
#         #             ret[a] = channel_ret[a]
#     # print("ret shape is: ", ret.shape)
#     return ret     # 3200 x 9


