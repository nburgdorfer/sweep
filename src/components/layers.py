from typing import Any

import torch.nn as nn
import torch.nn.functional as F
# from torchsparse import nn as spnn

#############################################
# 2D Convolution
#############################################
class Conv2d(nn.Module):
    def __init__(
            self,
            in_channels:int,
            out_channels:int,
            kernel_size:int|tuple[int,int]=3,
            stride:int|tuple[int,int]=1,
            padding:int|tuple[int,int]=1,
            normalization:dict[str, Any] | None={"type": "batch"},
            nonlinearity:dict[str, Any] | None={"type": "relu"}
        ):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=(normalization=="none"))

        if normalization is not None:
            if normalization["type"]=="batch":
                self.norm = nn.BatchNorm2d(out_channels)
            elif normalization["type"]=="group":
                self.norm = nn.GroupNorm(normalization["group_num"], out_channels)
            else:
                raise Exception(f"ERROR: Unknown normalization function: '{normalization["type"]}'")
        else:
            self.norm = None

        self.nonlinearity = nonlinearity

    def forward(self, x):
        out = self.conv(x)

        if self.norm is not None:
            out = self.norm(out)

        if self.nonlinearity is not None:
            if self.nonlinearity["type"] == "relu":
                out = F.relu(out)
            elif self.nonlinearity["type"] == "leaky_relu":
                out = F.leaky_relu(out)
            elif self.nonlinearity["type"] == "sigmoid":
                out = F.sigmoid(out)
            else:
                raise Exception(f"ERROR: Unknown nonlinearity function: '{self.nonlinearity["type"]}'")

        return out


#############################################
# 2D Deconvolution
#############################################
class ConvTranspose2d(nn.Module):
    def __init__(
            self,
            in_channels:int,
            out_channels:int,
            kernel_size:int|tuple[int,int]=3,
            stride:int|tuple[int,int]=1,
            padding:int|tuple[int,int]=1,
            output_padding:int|tuple[int,int]=1,
            normalization:dict[str, Any] | None={"type": "batch"},
            nonlinearity:dict[str, Any] | None={"type": "relu"}
        ):
        super(ConvTranspose2d, self).__init__()

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=(normalization=="none"))

        if normalization is not None:
            if normalization["type"]=="batch":
                self.norm = nn.BatchNorm2d(out_channels)
            elif normalization["type"]=="group":
                self.norm = nn.GroupNorm(normalization["group_num"], out_channels)
            else:
                raise Exception(f"ERROR: Unknown normalization function: '{normalization["type"]}'")
        else:
            self.norm = None

        self.nonlinearity = nonlinearity

    def forward(self, x):
        out = self.conv(x)

        if self.norm is not None:
            out = self.norm(out)

        if self.nonlinearity is not None:
            if self.nonlinearity["type"] == "relu":
                out = F.relu(out)
            elif self.nonlinearity["type"] == "leaky_relu":
                out = F.leaky_relu(out)
            elif self.nonlinearity["type"] == "sigmoid":
                out = F.sigmoid(out)
            else:
                raise Exception(f"ERROR: Unknown nonlinearity function: '{self.nonlinearity["type"]}'")

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
            # raise Exception(f"ERROR: Unknown normalization function: '{normalization}'")

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
            # raise Exception(f"ERROR: Unknown nonlinearity function: '{self.nonlinearity}'")

#         return out

#############################################
# 3D Convolution
#############################################
class Conv3d(nn.Module):
    def __init__(
            self,
            in_channels:int,
            out_channels:int,
            kernel_size:int|tuple[int,int,int]=3,
            stride:int|tuple[int,int,int]=1,
            padding:int|tuple[int,int,int]=1,
            normalization:dict[str, Any] | None={"type": "batch"},
            nonlinearity:dict[str, Any] | None={"type": "relu"}
    ):
        super(Conv3d, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=(normalization=="none"))

        if normalization is not None:
            if normalization["type"]=="batch":
                self.norm = nn.BatchNorm3d(out_channels)
            elif normalization["type"]=="group":
                self.norm = nn.GroupNorm(normalization["group_num"], out_channels)
            else:
                raise Exception(f"ERROR: Unknown normalization function: '{normalization["type"]}'")
        else:
            self.norm = None

        self.nonlinearity = nonlinearity

    def forward(self, x):
        out = self.conv(x)

        if self.norm is not None:
            out = self.norm(out)

        if self.nonlinearity is not None:
            if self.nonlinearity["type"] == "relu":
                out = F.relu(out)
            elif self.nonlinearity["type"] == "leaky_relu":
                out = F.leaky_relu(out)
            elif self.nonlinearity["type"] == "sigmoid":
                out = F.sigmoid(out)
            else:
                raise Exception(f"ERROR: Unknown nonlinearity function: '{self.nonlinearity["type"]}'")

        return out

#############################################
# 3D Deconvolution
#############################################
class ConvTranspose3d(nn.Module):
    def __init__(
            self,
            in_channels:int,
            out_channels:int,
            kernel_size:int|tuple[int,int,int]=3,
            stride:int|tuple[int,int,int]=1,
            padding:int|tuple[int,int,int]=1,
            output_padding:int|tuple[int,int,int]=1,
            normalization:dict[str, Any] | None={"type": "batch"},
            nonlinearity:dict[str, Any] | None={"type": "relu"}
    ):
        super(ConvTranspose3d, self).__init__()

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=(normalization=="none"))

        if normalization is not None:
            if normalization["type"]=="batch":
                self.norm = nn.BatchNorm3d(out_channels)
            elif normalization["type"]=="group":
                self.norm = nn.GroupNorm(normalization["group_num"], out_channels)
            else:
                raise Exception(f"ERROR: Unknown normalization function: '{normalization["type"]}'")
        else:
            self.norm = None

        self.nonlinearity = nonlinearity

    def forward(self, x):
        out = self.conv(x)

        if self.norm is not None:
            out = self.norm(out)

        if self.nonlinearity is not None:
            if self.nonlinearity["type"] == "relu":
                out = F.relu(out)
            elif self.nonlinearity["type"] == "leaky_relu":
                out = F.leaky_relu(out)
            elif self.nonlinearity["type"] == "sigmoid":
                out = F.sigmoid(out)
            else:
                raise Exception(f"ERROR: Unknown nonlinearity function: '{self.nonlinearity["type"]}'")

        return out

#############################################
# 3D Sparse Convolution
#############################################
class SparseConv3d(nn.Module):
    def __init__(
            self,
            in_channels:int,
            out_channels:int,
            kernel_size:int|tuple[int,int,int]=3,
            stride:int|tuple[int,int,int]=1,
            normalization:dict[str, Any] | None={"type": "batch"},
            nonlinearity:dict[str, Any] | None={"type": "relu"},
            transposed=False,
            factorize=True
    ):
        super(SparseConv3d, self).__init__()
        self.normalization = normalization
        self.nonlinearity = nonlinearity
        self.factorize = factorize
        bias = (normalization is None) or (normalization["type"]!="batch")

        if self.factorize:
            assert isinstance(kernel_size, int)
            assert isinstance(stride, int)
            conv1 = spnn.Conv3d(in_channels, in_channels, kernel_size=(1,1,kernel_size), stride=(0,0,stride), bias=bias, transposed=transposed)
            conv2 = spnn.Conv3d(in_channels, in_channels, kernel_size=(1,kernel_size,1), stride=(1,stride,1), bias=bias, transposed=transposed)
            conv3 = spnn.Conv3d(in_channels, out_channels, kernel_size=(kernel_size,1,1), stride=(stride,1,1), bias=bias, transposed=transposed)
            self.conv = nn.Sequential(nn.ModuleList([conv1, conv2, conv3]))
        else:
            self.conv = spnn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, bias=bias, transposed=transposed)

        if normalization is not None:
            if normalization["type"]=="batch":
                self.norm = nn.BatchNorm3d(out_channels)
            elif normalization["type"]=="group":
                self.norm = nn.GroupNorm(normalization["group_num"], out_channels)
            else:
                raise Exception(f"ERROR: Unknown normalization function: '{normalization["type"]}'")
        else:
            self.norm = None

        if nonlinearity is not None:
            if nonlinearity["type"]=="relu":
                self.activation = spnn.ReLU(True)
            else:
                raise Exception(f"ERROR: Unknown normalization function: '{nonlinearity["type"]}'")
        else:
            self.activation = None

    def forward(self, x):
        out = self.conv(x)

        if self.norm is not None:
            out = self.norm(out)

        if self.activation is not None:
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
    elif(nonlinearity!="none"):
        raise Exception(f"ERROR: Unknown nonlinearity function: '{nonlinearity}'")

    layers.append(nn.Linear(hidden_channels, out_channels, bias=bias))

    return nn.Sequential(*layers)