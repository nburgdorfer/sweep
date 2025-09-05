import torch
import torch.nn as nn
# from torchsparse import nn as spnn
# from torchsparse import SparseTensor

from src.components.layers import Conv3d, ConvTranspose3d, SparseConv3d

class LightRegularizer(nn.Module):
    def __init__(self, in_channels):
        super(LightRegularizer, self).__init__()

        self.conv0 = Conv3d(in_channels, 4)
        self.conv1 = Conv3d(4, 4)
        self.conv2 = Conv3d(4, 4)
        self.conv3 = Conv3d(4, 2)
        self.conv4 = Conv3d(2, 1, normalization=None, nonlinearity=None)

    def forward(self, x):
        out = self.conv4(self.conv3(self.conv2(self.conv1(self.conv0(x)))))

        return out

class BasicRegularizer(nn.Module):
    def __init__(self, in_channels, c):
        super(BasicRegularizer, self).__init__()

        self.conv0 = Conv3d(in_channels, c, padding=1)

        self.conv1 = Conv3d(  c, c*2, stride=2)
        self.conv2 = Conv3d(c*2, c*2)

        self.conv3 = Conv3d(c*2, c*4, stride=2)
        self.conv4 = Conv3d(c*4, c*4)

        self.conv5 = Conv3d(c*4, c*8, stride=2)
        self.conv6 = Conv3d(c*8, c*8)

        self.deconv7 = ConvTranspose3d(c*8, c*4, stride=2)
        self.deconv8 = ConvTranspose3d(c*4, c*2, stride=2)
        self.deconv9 = ConvTranspose3d(c*2, c*1, stride=2)

        self.prob = Conv3d(c, 1, normalization=None, nonlinearity=None)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.deconv7(x)
        x = conv2 + self.deconv8(x)
        x = conv0 + self.deconv9(x)
        x = self.prob(x)
        return x

class CostRegNet(nn.Module):
    def __init__(self, in_channels, c, out_channels=1):
        super(CostRegNet, self).__init__()
        self.conv0 = Conv3d(in_channels, c, padding=1)

        self.conv1 = Conv3d(c, c * 2, stride=(1, 2, 2), padding=1)
        self.conv2 = Conv3d(c * 2, c * 2, padding=1)

        self.conv3 = Conv3d(c * 2, c * 4, stride=(1, 2, 2), padding=1)
        self.conv4 = Conv3d(c * 4, c * 4, padding=1)

        self.conv5 = Conv3d(c * 4, c * 8, stride=(1, 2, 2), padding=1)
        self.conv6 = Conv3d(c * 8, c * 8, padding=1)

        self.deconv7 = ConvTranspose3d(c * 8, c * 4, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1))
        self.deconv8 = ConvTranspose3d(c * 4, c * 2, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1))
        self.deconv9 = ConvTranspose3d(c * 2, c * 1, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1))

        self.prob = nn.Conv3d(c, out_channels, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.deconv7(x)
        x = conv2 + self.deconv8(x)
        x = conv0 + self.deconv9(x)
        x = self.prob(x)
        return x


class DenseCostReg(nn.Module):
    def __init__(self,feature_ch):
        super(DenseCostReg, self).__init__()

        base_ch=feature_ch

        self.input0 = Conv3d(base_ch, base_ch, kernel_size=3, padding=1)
        self.input1 = Conv3d(base_ch, base_ch, kernel_size=3, padding=1)

        self.conv1a = Conv3d(base_ch, base_ch*2,stride=2, kernel_size=3, padding=1)
        self.conv1b = Conv3d(base_ch*2, base_ch*2, kernel_size=3, padding=1)
        self.conv1c = Conv3d(base_ch*2, base_ch*2, kernel_size=3, padding=1)
        self.conv2a = Conv3d(base_ch*2, base_ch*4,stride=2, kernel_size=3, padding=1)
        self.conv2b = Conv3d(base_ch*4, base_ch*4, kernel_size=3, padding=1)
        self.conv2c = Conv3d(base_ch*4, base_ch*4, kernel_size=3, padding=1)
        self.conv3a = Conv3d(base_ch*4, base_ch*8,stride=2, kernel_size=3, padding=1)
        self.conv3b = Conv3d(base_ch*8, base_ch*8, kernel_size=3, padding=1)
        self.conv3c = Conv3d(base_ch*8, base_ch*8, kernel_size=3, padding=1)

        self.conv3d = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear',align_corners=False),
            Conv3d(base_ch*8, base_ch*4,stride=1, kernel_size=3, padding=1)
        )

        self.conv2d = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear',align_corners=False),
            Conv3d(base_ch*4, base_ch*2,stride=1, kernel_size=3, padding=1)
        )

        self.conv1d = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear',align_corners=False),
            Conv3d(base_ch*2, base_ch,stride=1, kernel_size=3, padding=1)
        )

        self.prob0 = nn.Conv3d(base_ch, 1, 3, stride=1, padding=1)

    def forward(self, x, coord=None, mode="train"):

        input0 = self.input1(self.input0(x))

        conv1c = self.conv1c(self.conv1b(self.conv1a(input0)))
        conv2c = self.conv2c(self.conv2b(self.conv2a(conv1c)))
        conv3c = self.conv3c(self.conv3b(self.conv3a(conv2c)))
        
        conv3d = conv2c+self.conv3d(conv3c)
        conv2d = conv1c+self.conv2d(conv3d)
        conv1d = input0+self.conv1d(conv2d)

        prob = self.prob0(conv1d)

        return prob

class SparseCostReg(nn.Module):
    def __init__(self,feature_ch):
        super(SparseCostReg, self).__init__()

        base_ch=feature_ch

        self.input = nn.Sequential(
            SparseConv3d(base_ch, base_ch, kernel_size=3),
            SparseConv3d(base_ch, base_ch, kernel_size=3),
            SparseConv3d(base_ch, base_ch, kernel_size=3)
        )

        self.conv1up = nn.Sequential(
            SparseConv3d(base_ch, base_ch*2,stride=2, kernel_size=2, factorize=False),
            SparseConv3d(base_ch*2, base_ch*2, kernel_size=3),
            SparseConv3d(base_ch*2, base_ch*2, kernel_size=3)
        )
        self.conv2up = nn.Sequential(
            SparseConv3d(base_ch*2, base_ch*4,stride=2, kernel_size=2, factorize=False),
            SparseConv3d(base_ch*4, base_ch*4, kernel_size=3),
            SparseConv3d(base_ch*4, base_ch*4, kernel_size=3)
        )
        self.conv3up = nn.Sequential(
            SparseConv3d(base_ch*4, base_ch*8,stride=2, kernel_size=2, factorize=False),
            SparseConv3d(base_ch*8, base_ch*8, kernel_size=3),
            SparseConv3d(base_ch*8, base_ch*8, kernel_size=3)
        )

        self.conv3down = nn.Sequential(
            SparseConv3d(base_ch*8, base_ch*4, kernel_size=2, stride=2, factorize=False, transposed=True),
            SparseConv3d(base_ch*4, base_ch*4, kernel_size=3),
            SparseConv3d(base_ch*4, base_ch*4, kernel_size=3)
        )

        self.conv2down = nn.Sequential(
            SparseConv3d(base_ch*4, base_ch*2, kernel_size=2, stride=2, factorize=False, transposed=True),
            SparseConv3d(base_ch*2, base_ch*2, kernel_size=3),
            SparseConv3d(base_ch*2, base_ch*2, kernel_size=3)
        )

        self.conv1down = nn.Sequential(
            SparseConv3d(base_ch*2, base_ch, kernel_size=2, stride=2, factorize=False, transposed=True),
            SparseConv3d(base_ch, base_ch, kernel_size=3),
            SparseConv3d(base_ch, base_ch, kernel_size=3)
        )

        self.prob = nn.Sequential(
            SparseConv3d(base_ch, base_ch, kernel_size=3),
            SparseConv3d(base_ch, base_ch, kernel_size=3),
            spnn.Conv3d(base_ch, base_ch, (1,1,3), stride=1, bias=False),
            spnn.Conv3d(base_ch, base_ch, (1,3,1), stride=1, bias=False),
            spnn.Conv3d(base_ch, base_ch, (3,1,1), stride=1, bias=False),
            spnn.Conv3d(base_ch, 1, 1, stride=1, bias=False)
        )

    def forward(self, cost_volume, hypo_coords, mode='train'):
        # Convert cost volume and depth hypothesis to sparse feature and coordinates
        B,CH,D,H,W = cost_volume.shape
        # feats
        feats = cost_volume.permute(0,2,3,4,1).reshape(B*D*H*W,CH)
        # coords
        coords_z = hypo_coords.permute(0,2,3,4,1).reshape(B*D*H*W)
        coords_b, plain_coords_z, coords_h, coords_w = torch.where(torch.ones_like(hypo_coords.squeeze(1)))
        coords = torch.stack((coords_h,coords_w,coords_z,coords_b),dim=1).int()

        if mode == 'inference': 
            del coords_h,coords_w,coords_z,coords_b,plain_coords_z
            torch.cuda.empty_cache()

        # Make sparse feature
        x = SparseTensor(coords=coords, feats=feats)

        conv0 = self.input(x)

        if mode == "inference": 
            del x
            torch.cuda.empty_cache()

        conv1up = self.conv1up(conv0) # 1/2
        conv2up = self.conv2up(conv1up) # 1/4
        conv3up = self.conv3up(conv2up) # 1/8
        conv3down = conv2up+self.conv3down(conv3up) # 1/4

        if mode == "inference": 
            del conv3up
            del conv2up
            torch.cuda.empty_cache()

        conv2down = conv1up+self.conv2down(conv3down) # 1/2

        if mode == "inference": 
            del conv1up
            del conv3down
            torch.cuda.empty_cache()

        conv1down = conv0+self.conv1down(conv2down) # 1/1

        if mode == "inference": 
            del conv0
            del conv2down
            torch.cuda.empty_cache()

        prob = self.prob(conv1down)

        if mode == "inference": 
            del conv1down
            torch.cuda.empty_cache()

        # Convert back into dense volume
        est_prob = prob.F
        est_prob = est_prob.reshape((B,D,H,W,1))
        est_prob = est_prob.permute(0,4,1,2,3)

        return est_prob


class ViewWeightAgg(nn.Module):
    def __init__(self,va_feature_ch):
        super(ViewWeightAgg, self).__init__()

        base_ch = int(va_feature_ch/2)

        self.input0 = Conv3d(va_feature_ch, va_feature_ch, kernel_size=3, padding=1)
        self.input1 = Conv3d(va_feature_ch, base_ch, kernel_size=3, padding=1)

        self.conv1a = Conv3d(base_ch, base_ch*2,stride=2, kernel_size=3, padding=1)
        self.conv1b = Conv3d(base_ch*2, base_ch*2, kernel_size=3, padding=1)
        self.conv1c = Conv3d(base_ch*2, base_ch*2, kernel_size=3, padding=1)

        self.conv1d = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear',align_corners=False),
            Conv3d(base_ch*2, base_ch,stride=1, kernel_size=3, padding=1)
        )

        self.conv0 = nn.Sequential(
            Conv3d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.Conv3d(base_ch, 1, kernel_size=3, stride=1, padding=1, bias=False)
        )

        self.sigmoid = torch.sigmoid

    def forward(self, x):
        input0 = self.input1(self.input0(x))

        conv1c = self.conv1c(self.conv1b(self.conv1a(input0)))

        conv1d = input0+self.conv1d(conv1c)

        conv0 = self.conv0(conv1d)

        sig = self.sigmoid(conv0).squeeze(1)

        vis_weight, _ = torch.max(sig,dim=1)

        vis_threshold = 0.05

        vis_weight[vis_weight<vis_threshold] = 0

        return vis_weight.unsqueeze(1) # [B,1,H,W]


class PixelwiseNet(nn.Module):
    def __init__(self, in_channels):
        super(PixelwiseNet, self).__init__()
        self.conv0 = Conv3d(in_channels, 16, kernel_size=1, stride=1, padding=0)
        self.conv1 = Conv3d(16, 8, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv3d(8, 1, kernel_size=1, stride=1, padding=0)
        self.output = nn.Sigmoid()
        
    def forward(self, x1):
        x1 =self.conv2(self.conv1(self.conv0(x1)))
        output = self.output(x1)
        del x1
        output = torch.max(output, dim=1)[0]
        return output
