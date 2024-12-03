import os, sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import cv2
import time
import tinycudann as tcnn
import torchvision.models as models
from mmcv.ops.deform_conv import DeformConv2dPack

from src.components.layers import Conv2d, Deconv2d, Conv3d, Deconv3d

class BasicEncoder(nn.Module):
    def __init__(self, in_channels, c, out_channels, decode=False):
        super(BasicEncoder, self).__init__()
        self.decode = decode

        # H x W
        conv0 = [Conv2d(in_channels=in_channels, out_channels=c, kernel_size=3, padding=1, normalization="group")]
        conv0.append(Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, normalization="group"))
        self.conv0 = nn.Sequential(*nn.ModuleList(conv0))

        # H/2 x W/2
        conv1 = [Conv2d(in_channels=c, out_channels=c*2, kernel_size=5, stride=2, padding=2, normalization="group")]
        conv1.append(Conv2d(in_channels=c*2, out_channels=c*2, kernel_size=3, padding=1, normalization="group"))
        conv1.append(Conv2d(in_channels=c*2, out_channels=c*2, kernel_size=3, padding=1, normalization="group"))
        self.conv1 = nn.Sequential(*nn.ModuleList(conv1))

        # H/4 x W/4
        conv2 = [Conv2d(in_channels=c*2, out_channels=c*4, kernel_size=5, stride=2, padding=2, normalization="group")]
        conv2.append(Conv2d(in_channels=c*4, out_channels=c*4, kernel_size=3, padding=1, normalization="group"))
        conv2.append(Conv2d(in_channels=c*4, out_channels=c*4, kernel_size=3, padding=1, normalization="group"))
        self.conv2 = nn.Sequential(*nn.ModuleList(conv2))

        if self.decode:
            self.conv3 = Conv2d(in_channels=c*2, out_channels=c*4, kernel_size=3, padding=1, normalization="group")
            self.conv4 = Conv2d(in_channels=c, out_channels=c*2, kernel_size=3, padding=1, normalization="group")
            self.out1 = Conv2d(in_channels=c*4, out_channels=c*2, kernel_size=1, padding=0, normalization="group")
            self.out2 = Conv2d(in_channels=c*2, out_channels=out_channels, kernel_size=1, padding=0, normalization="none", nonlinearity="none")
        else:
            self.out1 = Conv2d(in_channels=c*4, out_channels=out_channels, kernel_size=1, padding=0, normalization="none", nonlinearity="none")


    def forward(self, img):
        z0 = self.conv0(img) # [c x H x W]
        z1 = self.conv1(z0) # [c*2 x H/2 x W/2]
        z2 = self.conv2(z1) # [c*4 x H/4 x W/4]

        if self.decode:
            f1 = self.out1(F.interpolate(z2, scale_factor=2, mode="bilinear") + self.conv3(z1)) # [c*2 x H/2 x W/2]
            out = self.out2(F.interpolate(f1, scale_factor=2, mode="bilinear") + self.conv4(z0)) # [out_channels x H x W]
        else:
            out = self.out1(z2) # [out_channels x H x W]

        return out


class FPN_small(nn.Module):
    def __init__(self, in_channels, c, out_channels):
        super(FPN_small, self).__init__()

        # H x W
        conv0 = [Conv2d(in_channels=in_channels, out_channels=c, kernel_size=3, padding=1)]
        conv0.append(Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1))
        self.conv0 = nn.Sequential(*nn.ModuleList(conv0))

        # H/2 x W/2
        conv1 = [Conv2d(in_channels=c, out_channels=c*2, kernel_size=5, stride=2, padding=2)]
        conv1.append(Conv2d(in_channels=c*2, out_channels=c*2, kernel_size=3, padding=1))
        conv1.append(Conv2d(in_channels=c*2, out_channels=c*2, kernel_size=3, padding=1))
        self.conv1 = nn.Sequential(*nn.ModuleList(conv1))

        # H/4 x W/4
        conv2 = [Conv2d(in_channels=c*2, out_channels=c*4, kernel_size=5, stride=2, padding=2)]
        conv2.append(Conv2d(in_channels=c*4, out_channels=c*4, kernel_size=3, padding=1))
        conv2.append(Conv2d(in_channels=c*4, out_channels=c*4, kernel_size=3, padding=1))
        self.conv2 = nn.Sequential(*nn.ModuleList(conv2))

        # H/8 x W/8
        conv3 = [Conv2d(in_channels=c*4, out_channels=c*8, kernel_size=5, stride=2, padding=2)]
        conv3.append(Conv2d(in_channels=c*8, out_channels=c*8, kernel_size=3, padding=1))
        conv3.append(Conv2d(in_channels=c*8, out_channels=c*8, kernel_size=3, padding=1))
        self.conv3 = nn.Sequential(*nn.ModuleList(conv3))

        # H/4 x W/4
        self.conv4 = Conv2d(in_channels=c*4, out_channels=c*8, kernel_size=1, padding=0, normalization="none", nonlinearity="none")
        self.conv5 = Conv2d(in_channels=c*2, out_channels=c*4, kernel_size=1, padding=0, normalization="none", nonlinearity="none")
        self.conv6 = Conv2d(in_channels=c, out_channels=c*2, kernel_size=1, padding=0, normalization="none", nonlinearity="none")

        # Output
        self.out0 = DeformConv2dPack(in_channels=c*8, out_channels=out_channels[0], kernel_size=3,stride=1,padding=1,deform_groups=1)
        self.out1 = DeformConv2dPack(in_channels=c*8, out_channels=out_channels[1], kernel_size=3,stride=1,padding=1,deform_groups=1)
        self.out2 = DeformConv2dPack(in_channels=c*4, out_channels=out_channels[2], kernel_size=3,stride=1,padding=1,deform_groups=1)
        self.out3 = DeformConv2dPack(in_channels=c*2, out_channels=out_channels[3], kernel_size=3,stride=1,padding=1,deform_groups=1)

    def forward(self, img):
        z0 = self.conv0(img) # [H x W]
        z1 = self.conv1(z0) # [H/2 x W/2]
        z2 = self.conv2(z1) # [H/4 x W/4]
        z3 = self.conv3(z2) # [H/8 x W/8]

        f0 = self.out0(z3) # [H/8 x W/8]
        f1 = self.out1(F.interpolate(z3, scale_factor=2, mode="bilinear") + self.conv4(z2)) # [H/4 x W/4]
        f2 = self.out2(F.interpolate(z2, scale_factor=2, mode="bilinear") + self.conv5(z1)) # [H/2 x W/2]
        f3 = self.out3(F.interpolate(z1, scale_factor=2, mode="bilinear") + self.conv6(z0)) # [H x W]

        return (f0, f1, f2, f3)

class FPN_large(nn.Module):
    def __init__(self, in_channels, c, out_channels):
        super(FPN_large, self).__init__()

        # H x W
        conv0 = [Conv2d(in_channels=in_channels, out_channels=c, kernel_size=3, padding=1)]
        conv0.append(Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1))
        conv0.append(Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1))
        conv0.append(Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1))
        conv0.append(Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1))
        self.conv0 = nn.Sequential(*nn.ModuleList(conv0))

        # H/2 x W/2
        conv1 = [Conv2d(in_channels=c, out_channels=c*2, kernel_size=5, stride=2, padding=2)]
        conv1.append(Conv2d(in_channels=c*2, out_channels=c*2, kernel_size=3, padding=1))
        conv1.append(Conv2d(in_channels=c*2, out_channels=c*2, kernel_size=3, padding=1))
        conv1.append(Conv2d(in_channels=c*2, out_channels=c*2, kernel_size=3, padding=1))
        conv1.append(Conv2d(in_channels=c*2, out_channels=c*2, kernel_size=3, padding=1))
        self.conv1 = nn.Sequential(*nn.ModuleList(conv1))

        # H/4 x W/4
        conv2 = [Conv2d(in_channels=c*2, out_channels=c*4, kernel_size=5, stride=2, padding=2)]
        conv2.append(Conv2d(in_channels=c*4, out_channels=c*4, kernel_size=3, padding=1))
        conv2.append(Conv2d(in_channels=c*4, out_channels=c*4, kernel_size=3, padding=1))
        conv2.append(Conv2d(in_channels=c*4, out_channels=c*4, kernel_size=3, padding=1))
        conv2.append(Conv2d(in_channels=c*4, out_channels=c*4, kernel_size=3, padding=1))
        self.conv2 = nn.Sequential(*nn.ModuleList(conv2))

        # H/8 x W/8
        conv3 = [Conv2d(in_channels=c*4, out_channels=c*8, kernel_size=5, stride=2, padding=2)]
        conv3.append(Conv2d(in_channels=c*8, out_channels=c*8, kernel_size=3, padding=1))
        conv3.append(Conv2d(in_channels=c*8, out_channels=c*8, kernel_size=3, padding=1))
        conv3.append(Conv2d(in_channels=c*8, out_channels=c*8, kernel_size=3, padding=1))
        conv3.append(Conv2d(in_channels=c*8, out_channels=c*8, kernel_size=3, padding=1))
        self.conv3 = nn.Sequential(*nn.ModuleList(conv3))

        # H/4 x W/4
        self.conv4 = Conv2d(in_channels=c*4, out_channels=c*8, kernel_size=1, padding=0, normalization="none", nonlinearity="none")
        self.conv5 = Conv2d(in_channels=c*2, out_channels=c*4, kernel_size=1, padding=0, normalization="none", nonlinearity="none")
        self.conv6 = Conv2d(in_channels=c, out_channels=c*2, kernel_size=1, padding=0, normalization="none", nonlinearity="none")

        # Output
        self.out0 = DeformConv2dPack(in_channels=c*8, out_channels=out_channels[0], kernel_size=3,stride=1,padding=1,deform_groups=1)
        self.out1 = DeformConv2dPack(in_channels=c*8, out_channels=out_channels[1], kernel_size=3,stride=1,padding=1,deform_groups=1)
        self.out2 = DeformConv2dPack(in_channels=c*4, out_channels=out_channels[2], kernel_size=3,stride=1,padding=1,deform_groups=1)
        self.out3 = DeformConv2dPack(in_channels=c*2, out_channels=out_channels[3], kernel_size=3,stride=1,padding=1,deform_groups=1)

    def forward(self, img):
        z0 = self.conv0(img) # [H x W]
        z1 = self.conv1(z0) # [H/2 x W/2]
        z2 = self.conv2(z1) # [H/4 x W/4]
        z3 = self.conv3(z2) # [H/8 x W/8]

        f0 = self.out0(z3) # [H/8 x W/8]
        f1 = self.out1(F.interpolate(z3, scale_factor=2, mode="bilinear") + self.conv4(z2)) # [H/4 x W/4]
        f2 = self.out2(F.interpolate(z2, scale_factor=2, mode="bilinear") + self.conv5(z1)) # [H/2 x W/2]
        f3 = self.out3(F.interpolate(z1, scale_factor=2, mode="bilinear") + self.conv6(z0)) # [H x W]

        return (f0, f1, f2, f3)


###### Experimental ######
class PSVEncoder(nn.Module):
    def __init__(self, in_channels, c=4):
        super(PSVEncoder, self).__init__()

        self.conv0 = Conv3d(in_channels, c, kernel_size=(1,5,5), padding=(0,2,2))
        self.conv1 = Conv3d(c, c, kernel_size=(1,3,3), padding=(0,1,1))
        self.conv2 = Conv3d(c, c*2, kernel_size=(1,3,3), padding=(0,1,1), stride=(1,2,2))

        self.conv3 = Conv3d(c*2, c*2, kernel_size=(1,5,5), padding=(0,2,2))
        self.conv4 = Conv3d(c*2, c*2, kernel_size=(1,3,3), padding=(0,1,1))
        self.conv5 = Conv3d(c*2, c*4, kernel_size=(1,3,3), padding=(0,1,1), stride=(1,2,2))

        self.conv6 = Conv3d(c*4, c*4, kernel_size=(1,3,3), padding=(0,1,1))
        self.conv7 = Conv3d(c*4, c*4, kernel_size=(1,3,3), padding=(0,1,1))
        self.conv8 = Conv3d(c*4, c*4, kernel_size=(1,3,3), padding=(0,1,1), normalization="none", nonlinearity="none")

    def forward(self, x):
        x = self.conv1(self.conv0(x)) # B x c x D x H x W
        x = self.conv4(self.conv3(self.conv2(x))) # B x c*2 x D x H/2 x W/2
        x = self.conv8(self.conv7(self.conv6(self.conv5(x)))) # B x c*4 x D x H/4 x W/4
        return x


class PSVEncoder_up(nn.Module):
    def __init__(self, in_channels, c=4):
        super(PSVEncoder_up, self).__init__()

        self.conv0 = Conv3d(in_channels, c, kernel_size=(1,5,5), padding=(0,2,2))
        self.conv1 = Conv3d(c, c, kernel_size=(1,3,3), padding=(0,1,1))
        self.conv2 = Conv3d(c, c*2, kernel_size=(1,3,3), padding=(0,1,1), stride=2)

        self.conv3 = Conv3d(c*2, c*2, kernel_size=(1,5,5), padding=(0,2,2))
        self.conv4 = Conv3d(c*2, c*2, kernel_size=(1,3,3), padding=(0,1,1))
        self.conv5 = Conv3d(c*2, c*4, kernel_size=(1,3,3), padding=(0,1,1), stride=2)

        self.conv6 = Conv3d(c*4, c*4, kernel_size=(1,3,3), padding=(0,1,1))

        self.deconv7 = Deconv3d(c * 4, c * 4, kernel_size=(1,3,3), stride=2, padding=(0,1,1), output_padding=1)
        self.deconv8 = Deconv3d((c*4) + (c*2), c * 4, kernel_size=(1,3,3), stride=2, padding=(0,1,1), output_padding=1)
        self.conv9 = Conv3d((c*4)+c, c * 4, kernel_size=(1,3,3), padding=(0,1,1))
        self.conv10 = Conv3d((c*4)+in_channels, c * 4, kernel_size=(1,3,3), padding=(0,1,1), normalization="none", nonlinearity="none")

    def forward(self, x):
        z0 = self.conv1(self.conv0(x)) # B x 8 x H x W
        z2 = self.conv4(self.conv3(self.conv2(z0))) # B x 16 x H/2 x W/2
        z3 = self.conv6(self.conv5(z2)) # B x 32 x H/4 x W/4

        z4 = self.deconv7(z3) # B x 32 x H/2 x W/2
        del z3

        z4 = torch.cat((z4,z2), dim=1)
        z5 = self.deconv8(z4) # B x 32 x H x W
        del z2
        del z4

        z5 = torch.cat((z5,z0), dim=1)
        z6 = self.conv9(z5) # B x 32 x H x W
        del z0
        del z5

        z6 = torch.cat((z6,x), dim=1)
        out = self.conv10(z6) # B x 32 x H x W
        del x
        del z6

        return out

class MLP_Mixer(nn.Module):
    def __init__(self, shape, h1, h2):
        super(MLP_Mixer, self).__init__()

        self.norm = nn.LayerNorm(shape, elementwise_affine=False)
        self.mlp_1 = mlp(shape[-2], h1, shape[-2], bias=False)
        self.mlp_2 = mlp(shape[-1], h2, shape[-1], bias=False)


    def forward(self, x):
        feats_1 = self.norm(x)
        feats_1 = feats_1.transpose(-2, -1)
        feats_1 = self.mlp_1(feats_1)
        feats_1 = feats_1.transpose(-2, -1)

        feats_1 = torch.add(feats_1, x)
        feats_2 = self.norm(feats_1)
        feats_2 = self.mlp_2(feats_2)
        output = torch.add(feats_1, feats_2)

        return output

