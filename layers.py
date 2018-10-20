import torch
from torch import nn
from torch.nn.functional import interpolate
from torch.nn import functional as F

class sSELayer(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.conv_1x1 = nn.Conv2d(n_in, 1, 1)

    def forward(self, x):
        desc = self.conv_1x1(x).sigmoid_()
        return desc * x

class scSELayer(nn.Module):
    def __init__(self, n_in, r = 16):
        assert n_in % r == 0, f'in channel count needs to be divisible by r == {r}'
        super().__init__()
        self.lin1 = nn.Linear(n_in, n_in // r)
        self.lin2 = nn.Linear(n_in // r, n_in)
        self.conv_1x1 = nn.Conv2d(n_in, 1, 1)

        self.bn_lin = nn.BatchNorm1d(n_in // r)
        self.bn_out = nn.BatchNorm2d(n_in)

    def forward(self, x):
        bs = x.shape[0]
        means = x.view(bs, x.shape[1], -1).mean(dim=-1)
        desc_c = F.relu(self.lin1(means))
        desc_c = self.bn_lin(desc_c)
        desc_c = self.lin2(desc_c).sigmoid_()

        desc_s = self.conv_1x1(x).sigmoid_()
        desc = desc_c[:,:,None,None] + desc_s
        return self.bn_out(desc * x)

class HCBlock(nn.Module):
    '''Hypercolumn block - reduces num of channels and interpolates'''
    def __init__(self, n_in, out_sz=256):
        super().__init__()
        self.conv = nn.Conv2d(n_in, 16, 1)
        self.bn = nn.BatchNorm2d(16)
        self.out_sz = out_sz

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.bn(x)
        return interpolate(x, (self.out_sz, self.out_sz), mode='bilinear', align_corners=False)
