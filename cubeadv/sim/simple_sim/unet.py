import torch
import torch.nn as nn
import torch.nn.functional as F

def double_conv(in_ch, out_ch):
    return nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, 3, padding=1),
                         nn.ReLU(inplace=True),
                         nn.Conv2d(out_ch, out_ch, 3, padding=1),
                         nn.ReLU(inplace=True))

class UNet(nn.Module):
    def __init__(self, inp_dim, height, width):
        super().__init__()
        self.height = height
        self.width = width
        self.start_height = height // 2**4 + 1
        self.start_width = width // 2**4 + 1
        self.inp_dim = inp_dim
        self.fc1 = nn.Linear(inp_dim,
            self.start_height * self.start_width * 512) # [*, 512, h0, w0]
        self.uc1 = nn.ConvTranspose2d(512, 512, 2, 2) # [*, 512, 2*h0, 2*w0]
        self.c1 = double_conv(512,  256)
        self.uc2 = nn.ConvTranspose2d(256, 256, 2, 2) # [*, 256, 4, 4]
        self.c2 = double_conv(256, 128)
        self.uc3 = nn.ConvTranspose2d(128, 128, 2, 2) # [*, 128, 8, 8]
        self.c3 = double_conv(128, 64)
        self.uc4 = nn.ConvTranspose2d(64, 64, 2, 2) # [*, 64, 16, 16]
        self.c4 = double_conv(64, 1)

    def query(self, x, c):
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if c.ndim == 1:
            c = c.unsqueeze(0)
        return None, self.forward(x, c)

    def forward(self, x, c):
        x = torch.cat([x.view(x.shape[0], -1), c], dim=1)
        x = F.relu(self.fc1(x))
        x = x.view(-1, 512, self.start_height, self.start_width)
        x = self.c1(self.uc1(x))
        x = self.c2(self.uc2(x))
        x = self.c3(self.uc3(x))
        x = self.c4(self.uc4(x))
        x = x[:, :, self.height, self.width].squeeze()

        return x
