import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsparse import nn as spnn

from torchvision import ops

#############################################
# 2D Convolution
#############################################
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, normalization="batch", nonlinearity="leaky_relu", group_num=2):
        super(Conv2d, self).__init__()
        self.normalization = normalization
        self.nonlinearity = nonlinearity

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=(normalization=="none"))

        if normalization=="batch":
            self.norm = nn.BatchNorm2d(out_channels)
        elif normalization=="group":
            self.norm = nn.GroupNorm(group_num, out_channels)
        elif normalization!="none":
            print(f"ERROR: Unknown normalization function: '{normalization}'")
            sys.exit(-1)

    def forward(self, x):
        out = self.conv(x)

        if self.normalization != "none":
            out = self.norm(out)

        if self.nonlinearity == "relu":
            out = F.relu(out)
        elif self.nonlinearity == "leaky_relu":
            out = F.leaky_relu(out)
        elif self.nonlinearity == "sigmoid":
            out = F.sigmoid(out)
        elif self.nonlinearity != "none":
            print(f"ERROR: Unknown nonlinearity function: '{self.nonlinearity}'")
            sys.exit()

        return out


#############################################
# 2D Deconvolution
#############################################
class Deconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=1, normalization="batch", nonlinearity="relu", group_num=8):
        super(Deconv2d, self).__init__()
        self.normalization = normalization
        self.nonlinearity = nonlinearity

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=(normalization=="none"))

        if normalization=="batch":
            self.norm = nn.BatchNorm2d(out_channels)
        elif normalization=="group":
            self.norm = nn.GroupNorm(group_num, out_channels)
        elif normalization!="none":
            print(f"ERROR: Unknown normalization function: '{normalization}'")
            sys.exit(-1)

    def forward(self, x):
        out = self.conv(x)

        if self.normalization != "none":
            out = self.norm(out)

        if self.nonlinearity != "none":
            out = F.relu(out)

        return out
    

# #############################################
# # 2D Deformable Convolution
# #############################################
# class DeformConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, normalization="none", nonlinearity="none", group_num=2):
#         super(DeformConv2d, self).__init__()
#         self.normalization = normalization
#         self.nonlinearity = nonlinearity

#         self.conv = ops.DeformConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=(normalization=="none"))
#         self.weights = 

#         if normalization=="batch":
#             self.norm = nn.BatchNorm2d(out_channels)
#         elif normalization=="group":
#             self.norm = nn.GroupNorm(group_num, out_channels)
#         elif normalization!="none":
#             print(f"ERROR: Unknown normalization function: '{normalization}'")
#             sys.exit(-1)

#     def forward(self, x):
#         out = self.conv(x)

#         if self.normalization != "none":
#             out = self.norm(out)

#         if self.nonlinearity == "relu":
#             out = F.relu(out)
#         elif self.nonlinearity == "leaky_relu":
#             out = F.leaky_relu(out)
#         elif self.nonlinearity == "sigmoid":
#             out = F.sigmoid(out)
#         elif self.nonlinearity != "none":
#             print(f"ERROR: Unknown nonlinearity function: '{self.nonlinearity}'")
#             sys.exit()

#         return out

#############################################
# 3D Convolution
#############################################
class Conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, normalization="batch", nonlinearity="relu", group_num=8):
        super(Conv3d, self).__init__()
        self.normalization = normalization
        self.nonlinearity = nonlinearity

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=(normalization=="none"))

        if normalization=="batch":
            self.norm = nn.BatchNorm3d(out_channels)
        elif normalization=="group":
            self.norm = nn.GroupNorm(group_num, out_channels)
        elif normalization!="none":
            print(f"ERROR: Unknown normalization function: '{normalization}'")
            sys.exit(-1)

    def forward(self, x):
        out = self.conv(x)

        if self.normalization != "none":
            out = self.norm(out)

        if self.nonlinearity != "none":
            out = F.relu(out)

        return out

#############################################
# 3D Deconvolution
#############################################
class Deconv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=1, normalization="batch", nonlinearity="relu", group_num=8):
        super(Deconv3d, self).__init__()
        self.normalization = normalization
        self.nonlinearity = nonlinearity

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=(normalization=="none"))

        if normalization=="batch":
            self.norm = nn.BatchNorm3d(out_channels)
        elif normalization=="group":
            self.norm = nn.GroupNorm(group_num, out_channels)
        elif normalization!="none":
            print(f"ERROR: Unknown normalization function: '{normalization}'")
            sys.exit(-1)

    def forward(self, x):
        out = self.conv(x)

        if self.normalization != "none":
            out = self.norm(out)

        if self.nonlinearity != "none":
            out = F.relu(out)

        return out

#############################################
# 3D Sparse Convolution
#############################################
class SparseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, normalization="batch", nonlinearity="relu", transposed=False, bias=False, factorize=True):
        super(SparseConv3d, self).__init__()
        self.normalization = normalization
        self.nonlinearity = nonlinearity
        self.factorize = factorize

        if self.factorize:
            self.conv1 = spnn.Conv3d(in_channels, in_channels, kernel_size=(1,1,kernel_size), stride=stride, bias=(normalization!="batch"), transposed=transposed)
            self.conv2 = spnn.Conv3d(in_channels, in_channels, kernel_size=(1,kernel_size,1), stride=stride, bias=(normalization!="batch"), transposed=transposed)
            self.conv3 = spnn.Conv3d(in_channels, out_channels, kernel_size=(kernel_size,1,1), stride=stride, bias=(normalization!="batch"), transposed=transposed)
        else:
            self.conv = spnn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, bias=(normalization!="batch"), transposed=transposed)

        if normalization=="batch":
            self.norm = spnn.BatchNorm(out_channels)
        elif normalization!="none":
            print(f"ERROR: Unknown normalization function: '{normalization}'")
            sys.exit(-1)

        if (nonlinearity=="relu"):
            self.activation = spnn.ReLU(True)
        elif normalization!="none":
            print(f"ERROR: Unknown nonlinearity function: '{nonlinearity}'")
            sys.exit(-1)

    def forward(self, x):
        if self.factorize:
            out = self.conv3(self.conv2(self.conv1(x)))
        else:
            out = self.conv(x)

        if self.normalization != "none":
            out = self.norm(out)

        if self.nonlinearity != "none":
            out = self.activation(out)

        return out


#############################################
# MLP
#############################################
def mlp(in_channels, hidden_channels, out_channels, bias=False, nonlinearity="relu"):
    layers = []

    layers.append(nn.Linear(in_channels, hidden_channels, bias=bias))

    if (nonlinearity=="relu"):
        layers.append(nn.ReLU(True))
    elif (nonlinearity=="leaky_relu"):
        layers.append(nn.LeakyReLU(True))
    elif (nonlinearity=="sigmoid"):
        layers.append(nn.Sigmoid(True))
    elif (nonlinearity=="tanh"):
        layers.append(nn.Tanh(True))
    elif (nonlinearity=="softmax"):
        layers.append(nn.Softmax(True))
    elif (nonlinearity=="gelu"):
        layers.append(nn.GELU(True))
    elif(nonlinearity!="none"):
        print("ERROR: Unkown nonlinearity function: '{}'".format(nonlinearity))
        sys.exit(-1)

    layers.append(nn.Linear(hidden_channels, out_channels, bias=bias))

    return nn.Sequential(*layers)