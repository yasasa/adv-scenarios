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
                x = os[idx]

                y = us[idx]
                yt = self._net(x)
#                 print("output size:", yt.size())
#                 print("target size:", y.size())
                
                loss = torch.sum((y - yt)**2, dim=1).mean()
#                 print("loss size:", loss.size())

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
        self.fc5 = nn.Linear(80, 2)

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
        self.fc5 = nn.Linear(80, 2)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        return x


def loss_function(pred, target):
    #MSE = F.l1_loss(pred, target, reduction='sum')      # default: reduction='mean'
    MSE = F.mse_loss(pred, target, reduction='sum')

    return MSE
