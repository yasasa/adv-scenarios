import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import Resize, CenterCrop
from torchvision.ops import Permute

class Policy_(nn.Module):
    def __init__(self, nc=3):
        super().__init__()
        self.policies = nn.ModuleList([PolicyInternal() for policy in range(nc)])

    def forward(self, o, goal):
        us = []
        for module in self.policies:
            u = module(o, goal)
            us.append(u)
        masked = torch.cat(us, dim=-1)[goal==1]
        return masked.unsqueeze(-1)

def get_img_transform():
    transform = torch.nn.Sequential(
            Permute([2, 0, 1]), # HWC -> CHW
#            Resize((145, 320), antialias=True),
            CenterCrop((66, 200)),
            )
    return transform


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
       # self.norm = nn.BatchNorm2d(3)
        self.norm = lambda x: x
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2),
            nn.ReLU(),
            nn.Conv2d(24, 36, 5, 2),
            nn.ReLU(),
            nn.Conv2d(36, 48, 5, 2),
            nn.ReLU(),
            nn.Conv2d(48, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU()
        )


        self.linear = nn.Sequential(
                       # nn.Linear(35840, 1000),
                        nn.Linear(64*18, 1000),
                        nn.ReLU(),
                        nn.Linear(1000, 256),
                        nn.ReLU(),
                        nn.Linear(256, 256),
                        nn.ReLU(),
                        nn.Linear(256, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1)
                      )

    def forward(self, image):
        img_features = self.conv(self.norm(image))
        img_features = img_features.reshape(image.shape[0], -1)
        x = self.linear(img_features)
        return torch.tanh(x)

    @property
    def requested_feature_shape(self):
        return (3, 66, 200)
        
class PolicyDepth(nn.Module):        
    def __init__(self):
        super().__init__()
       # self.norm = nn.BatchNorm2d(3)
        self.norm = lambda x: x
        self.conv = nn.Sequential(
            nn.Conv2d(4, 24, 5, 2),
            nn.ReLU(),
            nn.Conv2d(24, 36, 5, 2),
            nn.ReLU(),
            nn.Conv2d(36, 48, 5, 2),
            nn.ReLU(),
            nn.Conv2d(48, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU()
        )


        self.linear = nn.Sequential(
                        nn.Linear(64*18 + 3, 1000),
                        nn.ReLU(),
                        nn.Linear(1000, 256),
                        nn.ReLU(),
                        nn.Linear(256, 256),
                        nn.ReLU(),
                        nn.Linear(256, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1)
                      )

    def forward(self, image, goal, depth, start=None):
        image = torch.cat([image, depth], dim=1)
        img_features = self.conv(self.norm(image))
        img_features = img_features.reshape(image.shape[0], -1)
        linear_features = torch.cat([img_features, goal], dim=-1)
        x = self.linear(linear_features)
        return torch.tanh(x)

    @property
    def requested_feature_shape(self):
        return (4, 66, 200)

class PolicyNoStart(nn.Module):
    def __init__(self):
        super().__init__()
       # self.norm = nn.BatchNorm2d(3)
        self.norm = lambda x: x
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2),
            nn.ReLU(),
            nn.Conv2d(24, 36, 5, 2),
            nn.ReLU(),
            nn.Conv2d(36, 48, 5, 2),
            nn.ReLU(),
            nn.Conv2d(48, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU()
        )


        self.linear = nn.Sequential(
                        nn.Linear(64*18 + 3, 1000),
                        nn.ReLU(),
                        nn.Linear(1000, 256),
                        nn.ReLU(),
                        nn.Linear(256, 256),
                        nn.ReLU(),
                        nn.Linear(256, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1)
                      )

    def forward(self, image, goal, depth=None, start=None):
        img_features = self.conv(self.norm(image))
        img_features = img_features.reshape(image.shape[0], -1)
        linear_features = torch.cat([img_features, goal], dim=-1)
        x = self.linear(linear_features)
        return torch.tanh(x)

    @property
    def requested_feature_shape(self):
        return (3, 66, 200)

