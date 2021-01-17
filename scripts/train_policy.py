import torch

from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import numpy as np

from cubeadv.policies.cnn import get_img_transform, Policy

import imageio

import torchvision
import torchvision.transforms as T
from torchvision.datasets import VisionDataset
import json
import os

import argparse
import wandb

class DrivingDataset(Dataset):
    def __init__(self, path, transform):
        super().__init__()
        self.path = path
        with open(os.path.join(self.path, "controls.json"), "r") as f:
            self.meta = json.load(f)
        self.files = self.meta.keys()
        self.transform = transform
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        filename = f"{i:05d}.jpg"
        u = self.meta[filename]["ctl"]
        im = imageio.imread(os.path.join(self.path, filename))
        im = torch.FloatTensor(im) / 255.
        if self.transform is not None:
            im = self.transform(im)

        return im, u
 
@torch.no_grad()
def val(data_itr, model, loss=torch.nn.functional.mse_loss, device="cuda:0"):
    total_loss = 0
    tl = len(data_itr)
    model.eval()
    for itr, imu in enumerate(data_itr):
        im, u = imu
        sample_image = wandb.Image(im[0].permute(1, 2, 0).numpy())
        sample_gt = u[0].numpy()
        im = im.to(device)
        us = u.type_as(im)

        
        yh = model(im).squeeze()
        if yh.dim() >= 1:
            sample_out = yh[0].detach().cpu().numpy()
        else:
            sample_out = yh.detach().cpu().numpy()
        
        l = loss(yh, us)

        total_loss += l.item()
        wandb.log({"val_itr": itr, 
                   "sample_image": sample_image, 
                   "sample_ctl": sample_out, 
                   "sample_gt": sample_gt})

    return total_loss / tl

def train_epoch(data_itr, optimizer, model, loss=torch.nn.functional.mse_loss, device="cuda:0", logging_cb=None):
    total_loss = 0
    for itr, imu in enumerate(data_itr):
        im, u = imu
        im = im.to(device)
        us = u.type_as(im)
        
        optimizer.zero_grad()
        yh = model(im).squeeze()
        l = loss(yh, us)
        l.backward()
        optimizer.step()

        total_loss += l.item()
        logging_cb(itr, total_loss, model)

def logging(epoch, itr, loss, model):
    wandb.log({
        "epoch" : epoch,
        "batch" : itr,
        "training_loss" : loss / (itr + 1),
    })

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", type=str, required=True)
    parser.add_argument("--training-data", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--train-split", type=int, default=80)


    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    wandb.init(project="my580-rc-car-driving", config=args)
    torch.manual_seed(args.seed)
    
    model = Policy().to(args.device)
    opt = torch.optim.Adam(model.parameters())

    training_transform = torch.nn.Sequential(
                    get_img_transform(),
                    #T.ColorJitter(contrast=0.9, hue=0.3, brightness=0.5),
                   # T.RandomErasing(),
                    T.RandomPerspective(distortion_scale=0.1, p=0.1)
                )
    
    train_set = DrivingDataset(args.training_data, training_transform)
    val_set = DrivingDataset(args.test_data, transform=get_img_transform())

    big_set = ConcatDataset([train_set, val_set])
    train_size = int(len(train_set) * args.train_split / 100)
    test_size = len(train_set) - train_size
    train_dataset, test_dataset = random_split(train_set, [train_size, test_size]) 
    itr = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_itr = DataLoader(test_dataset, batch_size=2, shuffle=True, pin_memory=True)

    best_val = np.inf
    for epoch in range(args.epochs):
        log_cb = lambda i, l, m: logging(epoch, i, l, m)
        train_epoch(itr, opt, model, device=args.device, logging_cb=log_cb)
        validation = val(val_itr, model, device=args.device)
        if validation < best_val:
            best_val = validation
            torch.save(model.state_dict(), "policy.ckpt")
        wandb.log({"epoch" : epoch, "val" : validation})
    
