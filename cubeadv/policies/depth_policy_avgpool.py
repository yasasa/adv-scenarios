"""
This is a CNN implementation without variational inference
forked from VCNN2

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import cv2
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

import random
# from load_data import load_data
# from RGBDataset import RGBDataset
import os
#conda activate f_frcnn_py36

class Policy:
    def __init__(self, net, rgb=False):
        self._net = net
        self.channels = 4 if rgb else 1
        self.avgpool = torch.nn.AvgPool2d(2,stride=2)

    def _pi(self, o, tensor=False):
        self._net = self._net.eval()

        if not tensor:
            o = torch.tensor(o).float()


        if o.dim()==3:
            o = o.unsqueeze(0)

        o = o.reshape(-1, 3200, self.channels)
        o = o.view(-1, 32, 100, self.channels)
        o = o.permute(0, 3, 1, 2)
        y = self._net(o.cuda())
        if not tensor:
            y = y.detach().cpu().numpy().flatten()
        return y

    def pi(self, o, tensor=False):
        self._net = self._net.eval()

        if not tensor:
            o = torch.tensor(o).float()
        
        # o.size = [200, 64, 4]
        o = self.avgpool(o.permute(2, 0, 1).unsqueeze(0)).squeeze(0).permute(1, 2, 0)
        o = o.reshape(-1, 3200 * self.channels)
        
        y = self._net(o.cuda())
        if not tensor:
            y = y.detach().cpu().numpy().flatten()
        return y.squeeze()



    def _J(self, o):
        shape = o.shape
        o_ = o.flatten()
        J = np.zeros((1, o_.shape[0]))
        for i in range(o_.shape[0]):
            e = np.eye(o_.shape[0])[i]*1e-6
            o1 = o_ + e
            o2 = o_ - e
            J[:, i] = (self.pi(o1) - self.pi(o2)) / 2e-6

        return J

    def J(self, o):
        o = torch.tensor(o, requires_grad=True, dtype=torch.float32)
        y = self.pi(o, tensor=True)
        J = torch.autograd.grad(y, o)[0]
        J = J.reshape(1, -1)

        return J.detach().cpu().numpy()

    def __call__(self, o):
        return self.pi(o, tensor=torch.is_tensor(o))

    def train(self, os, us, steps=10):
        optimizer = torch.optim.Adam(self._net.parameters(), lr=1e-6)

        datalen = os.shape[0]
        bs = 50
        indices = torch.randperm(datalen)
        batch_indices = torch.split(indices, bs)

        self._net.train()

        for i in range(steps):
            rl = 0.
            for idx in batch_indices:
                optimizer.zero_grad()
                x = torch.from_numpy(os[idx]).float().cuda()
                y = torch.from_numpy(us[idx]).float().cuda()
                yt = self._net(x)
                loss = ((y - yt)**2).mean(0)
                loss.backward()
                optimizer.step()
                rl += loss.detach().cpu().item()


            l = rl / len(batch_indices)
            print("[Training Step {:d}] Loss {:.4f}".format(i+1, l))

        return l

    def save(self, path):
        torch.save(self._net.state_dict(), path)

class RGBNet(nn.Module):

    def __init__(self):
        super(RGBNet, self).__init__()
        self.fc1 = nn.Linear(3200*4, 640*4)
        self.fc2 = nn.Linear(640*4, 320)
        self.fc3 = nn.Linear(320, 160)
        self.fc4 = nn.Linear(160, 80)
        self.fc5 = nn.Linear(80, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        return x

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3200, 640)
        self.fc2 = nn.Linear(640, 320)
        self.fc3 = nn.Linear(320, 160)
        self.fc4 = nn.Linear(160, 80)
        self.fc5 = nn.Linear(80, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        return x

class _Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # kernel
        self.conv1 = nn.Conv2d(1, 32, 3)    # 180 * 180 * 64
        self.conv2 = nn.Conv2d(32, 32, 3)   # 90
        self.conv3 = nn.Conv2d(32, 32, 3)   # 45
        #self.conv4 = nn.Conv2d(64, 64, 3)   # 9
        #self.conv5 = nn.Conv2d(64, 64, 3)   # 3

        # an affine operation: y = Wx + b
        # self.fc11 = nn.Linear(448, 240)  # 6*6 from image dimension
        # self.fc12 = nn.Linear(448, 240)  # 6*6 from image dimension
        # self.fc1 = nn.Linear(448, 240)  # 6*6 from image dimension
        self.fc1 = nn.Linear(640, 240)  # 6*6 from image dimension
        self.fc2 = nn.Linear(240, 128)
        self.fc3 = nn.Linear(128, 1)


    def encode(self, x):
        # Max pooling over a (2, 2) window
        # print(x.shape)
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(x), (2, 2)) # 90 * 90 * 64
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 45 * 45 * 64
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)  # 9 * 9 * 64
        x = x.view(-1, self.num_flat_features(x))
        return x

    def decode(self, z):
        z = F.relu(self.fc1(z))
        y = F.relu(self.fc2(z))
        return self.fc3(y)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def loss_function(pred, target):
    #MSE = F.l1_loss(pred, target, reduction='sum')      # default: reduction='mean'
    MSE = F.mse_loss(pred, target, reduction='sum')

    return MSE
