import torch
import torch.nn as nn

from src.components.layers import Conv2d


class BasicRefiner(nn.Module):
    def __init__(self, in_channels, c):
        super(BasicRefiner, self).__init__()
        self.conv1 = Conv2d(in_channels, c)
        self.conv2 = Conv2d(c, c)
        self.conv3 = Conv2d(c, c)
        self.res = Conv2d(c, 1)

    def forward(self, img, depth_init):
        concat = torch.cat((img, depth_init), dim=1)
        depth_residual = self.res(self.conv3(self.conv2(self.conv1(concat))))
        depth_refined = depth_init + depth_residual
        return depth_refined
