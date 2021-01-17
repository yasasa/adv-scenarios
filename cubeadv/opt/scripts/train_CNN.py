"""
Training code for variational point fusion network
VPF

"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import cv2
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

import random

from CNN import Net, loss_function

## data loader
from data_processors import *
from PointFusion_dataset import LidarOnlyDataset


## arg parser
import argparse
argparser = argparse.ArgumentParser(
        description='training LidarCNN using Dagger')
## dagger iteration number    
argparser.add_argument(
    '--iter',
    default='-1',
    help='dagger iteration number')
argparser.add_argument(
    '--expnum',
    default=0,
    type=int,
    help='number of experiment run today (default: 0)')
args = argparser.parse_args()

#############################################################################
## parameters
batchSize = 100#100
npoints = 3200
nepoch = 50#100
can_load_pretrained = 1
load_pretrained = 0
#weights_path = '/home/zidong/Desktop/nn/pointFusion/6.12_2/cls_model_47.pth'
import datetime as dt
now = dt.datetime.now()
outf = 'experiments/%d-%d-%d.%d/model' % (now.year, now.month, now.day, args.expnum)

raw_data_path = 'experiments/%d-%d-%d.%d/data/ep_' % (now.year, now.month, now.day, args.expnum)
# this is the number of dagger iteration

iteration = int(args.iter)
assert(iteration>=0)

## set up the weight_path for pretrained model from last iteration
weights_path =  '%s/dagger_%d.pth' % (outf, iteration-1)
#filename = '711_vpf1'
#weights_path =  '%s/%s.pth' % (outf, filename)
#############################################################################
if can_load_pretrained and iteration > 0:
    load_pretrained = 1


## preproccesing
print("preproccesing..")

## reading image
## note: 100 pictures per batch
print('loading data from', raw_data_path)

## iteration+1
# data_points, data_labels = load_data_colour_lidar(raw_data_path, 0, iteration+1, 0, 20) # ep 0-3
data_points, data_labels = load_data_cnn_lidar(raw_data_path, 0, iteration+1, 0, 23) # ep 0-3
#tst_inputs, tst_points, tst_labels = load_data(raw_data_path, 4, 5, 0, 44)   # ep 4

## load data with points and labels

# dataset = RGBLidarDataset(
#     data_points=data_points,
#     data_labels=data_labels,
#     data_rgb = data_rgb)
dataset = LidarOnlyDataset(
    data_points=data_points,
    data_labels=data_labels)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchSize,
    shuffle=True,
    num_workers=4)

## note: test data is same as training - only first ep => PID data
test_points, test_labels = load_data_cnn_lidar(raw_data_path, 0, 1, 0, 23)
# test_rgb, test_points, test_labels = load_data(raw_data_path, 0, 1, 0, 20)
# dataset_test = RGBLidarDataset(
#     data_points=test_points,
#     data_labels=test_labels,
#     data_rgb = test_rgb)
dataset_test = LidarOnlyDataset(
    data_points=test_points,
    data_labels=test_labels)

testloader = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=batchSize,#batchSize,
    shuffle=False,
    num_workers=4)

print('finished loading data, label length =', len(data_labels), ', point length = ', len(data_points))

#############################################################################
## training

net = Net()

optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
net.cuda()

## loss = MSE + KL


num_batch = data_points.shape[0]  / batchSize
print('num_batch =', num_batch)

## load pretrained

if load_pretrained: 
    #'/home/zidong/Desktop/pointnet.pytorch/utils/cls/cls_model_19.pth'
    net.load_state_dict(torch.load(weights_path))
    print('loading pretrained model from ', weights_path)
else:
    print('no pretrained model, start training..')


train_loss = [] # total training loss = loss / num data
total_loss = []

for epoch in range(nepoch):
    scheduler.step()
    total_loss_ephoc = [0.0, 0.0, 0.0, 0.0]
    for i, data in enumerate(dataloader, 0):
        
        # images, points, target = data
        points, target = data
        ## actually 1900
        points = points.reshape(-1, 4, 32, 100)
        # print(points.shape)
        # images, points, target = images.cuda(), points.cuda(), target.cuda()
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()        
        net = net.train()

        pred = net(points)
        pred = torch.reshape(pred, (-1,))
        loss = loss_function(pred, target)
        
        total_loss_ephoc[0] += loss.detach().tolist() #* images.size(0)
        # total_loss_ephoc[1] += mse.detach().tolist()
        # total_loss_ephoc[2] += kl.detach().tolist()

        
        total_loss_ephoc[0] += loss.detach().tolist() #/ target.shape[0]
        
        loss.backward()
        optimizer.step()
        
        
        if i % 10 == 0:
            #print('prediction = ', pred)
            print('[%d: %d/%d] train loss: %f'  % (epoch, i, num_batch, loss.item()/batchSize)) # average loss
    
    total_loss_ephoc[0] /= data_points.shape[0]

    train_loss.append(total_loss_ephoc)
    
    if 1:#epoch % 10 == 0:
        running_loss = [0.0, 0.0, 0.0, 0.0]
        #'''
        for i, data in enumerate(testloader, 0):
            net = net.eval()
            # images_, points_, target_ = data
            points_, target_ = data
            points_ = points_.reshape(batchSize, 4, 32, 100)
            # images_, points_, target_ = images_.cuda(), points_.cuda(), target_.cuda()
            points_, target_ = points_.cuda(), target_.cuda()
            #print(inputs.shape)
            with torch.no_grad():
                pred_ = net(points_)
                pred_ = torch.reshape(pred_, (-1,))
                
                #loss = F.mse_loss(pred_, target_, reduction='sum')
                loss = loss_function(pred, target)
                
                running_loss[0] += loss.detach().tolist() #* images.size(0)
                # running_loss[1] += mse.detach().tolist()
                # running_loss[2] += kl.detach().tolist()

        total_loss.append(running_loss)
        running_loss = [x / test_points.shape[0] for x in running_loss]
        print('epoch %d running_loss is ' % (epoch), running_loss) # average loss
            # '''
torch.save(net.state_dict(), '%s/dagger_%d.pth' % (outf, iteration))
print('stored trained model in ', '%s/dagger_%d.pth' % (outf, iteration))


## same loss curve
# total_loss = np.asarray(total_loss)
# loss_log_name = '%s/dagger_%d_loss.pth' % (outf, iteration)
# np.save(loss_log_name, total_loss)

## save the total training loss
train_loss = np.asarray(train_loss)
loss_log_name2 = '%s/dagger_%d_loss_all.pth' % (outf, iteration)
np.save(loss_log_name2, total_loss)

#############################################################################
