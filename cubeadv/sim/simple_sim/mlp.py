import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, height, width):
        super().__init__()
        self.input_dim = input_dim
        self.height = height
        self.width = width

        out = height * width * 4
        inp = self.input_dim

        layers = [inp, 1, 1]

        module_list = []
        for i, o in zip(layers[:-1], layers[1:]):
            module_list.append(nn.Linear(i, o))
            module_list.append(nn.ReLU())

        module_list.append(nn.Linear(layers[-1], out))

        self._net = nn.Sequential(*module_list)

    def normalize(self, x, c):
        max = torch.tensor([120, 140.], device=x.device)
        min = torch.tensor([80., 110.], device=x.device)
        size = (max - min)

        locs = x[:, :2, 3]
        x[:, :2, 3] = (locs - min) / size
        c[:, :2] = (c[:, :2] - min) / size
        return x, c

    def run(self, x, c, props, *args, **kwargs):
        _c = torch.cat([c, props], dim=-1)
        return self.query(x, _c)

    def query(self, x, c):
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if c.ndim == 1:
            c = c.unsqueeze(0)
        return self.forward(*self.normalize(x, c))

    def query_batch(self, x, c):
        return self.query(x, c)[1]

    def forward(self, x, c):
        xc = torch.cat([x.view(x.shape[0], -1), c], dim=1)
        y = self._net.forward(xc)
        rgbd = y.view(y.shape[0], self.height, self.width, 4)
        return rgbd[:, :, :, :3].squeeze(), rgbd[:, :, :, 3].squeeze()


