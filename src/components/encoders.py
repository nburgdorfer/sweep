import torch
import torch.nn as nn
import torch.nn.functional as F

from src.components.layers import Conv2d, ConvTranspose2d, Conv3d, ConvTranspose3d, mlp

class BasicEncoder(nn.Module):
    def __init__(self, in_channels, c, out_channels, decode=False):
        super(BasicEncoder, self).__init__()
        self.decode = decode

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

        if self.decode:
            self.conv3 = Conv2d(in_channels=c*2, out_channels=c*4, kernel_size=3, padding=1)
            self.conv4 = Conv2d(in_channels=c, out_channels=c*2, kernel_size=3, padding=1)
            self.out1 = Conv2d(in_channels=c*4, out_channels=c*2, kernel_size=1, padding=0)
            self.out2 = Conv2d(in_channels=c*2, out_channels=out_channels, kernel_size=1, padding=0, normalization=None, nonlinearity=None)
        else:
            self.out1 = Conv2d(in_channels=c*4, out_channels=out_channels, kernel_size=1, padding=0, normalization=None, nonlinearity=None)


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


class FPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 base_channels=8,
                 block_size=3,
                 levels=4,
                 out_levels=4,
                 low_res_first=True
                 ):
        """
        Args:
            in_channels: The expected channel size for the input data.
            out_channels: The channel size for the output features.
            base_channels: The base hidden channel size inside the encoder.
            block_size: The number of convolution layers per convolution block.
            levels: The number of total pyramid layers.
            out_levels: The expected number of output feature maps.
            low_res_first: Specifies the resolution order of the output features list.
                if True, the list will be coarse->fine (element 0 will be the lowest resolution);
                if False, the list will be fine->coarse (element 0 will be the highest resolution)
        """
        super(FPN, self).__init__()

        assert block_size >= 1
        assert levels >= 1
        assert out_levels <= levels

        self.levels = levels
        self.out_levels = out_levels
        self.low_res_first = low_res_first

        # build hidden channels list
        hidden_channels = []
        for l in range(levels):
            hidden_channels.append(base_channels * (2**l))

        ### UP Convolution Layers ###
        self.up_conv = nn.ModuleList()
        conv_0 = [Conv2d(in_channels=in_channels, out_channels=hidden_channels[0], kernel_size=3, padding=1)]
        for _ in range (1, block_size):
            conv_0.append(Conv2d(in_channels=hidden_channels[0], out_channels=hidden_channels[0], kernel_size=3, padding=1))
        self.up_conv.append(nn.Sequential(*nn.ModuleList(conv_0)))
        
        for l in range(1, levels):
            conv = [Conv2d(in_channels=hidden_channels[l-1], out_channels=hidden_channels[l], kernel_size=5, stride=2, padding=2)]
            for _ in range (1, block_size):
                conv.append(Conv2d(in_channels=hidden_channels[l], out_channels=hidden_channels[l], kernel_size=3, padding=1))
            self.up_conv.append(nn.Sequential(*nn.ModuleList(conv)))

        ### Lateral Convolution Layers ###
        self.lateral_conv = nn.ModuleList()
        for l in range(levels-1):
            self.lateral_conv.append(Conv2d(in_channels=hidden_channels[l], out_channels=hidden_channels[l+1], kernel_size=1, padding=0))
        self.lateral_conv.append(Conv2d(in_channels=hidden_channels[self.levels-1], out_channels=hidden_channels[self.levels-1], kernel_size=1, padding=0))

        ### Down Convolution Layers ###
        self.down_conv = nn.ModuleList()
        for l in range(levels-1):
            conv = [Conv2d(in_channels=hidden_channels[l+1], out_channels=hidden_channels[l], kernel_size=3, padding=1)]
            for _ in range (1, block_size):
                conv.append(Conv2d(in_channels=hidden_channels[l], out_channels=hidden_channels[l], kernel_size=3, padding=1))
            self.down_conv.append(nn.Sequential(*nn.ModuleList(conv)))

        ### Output Convolution Layers ###
        self.out_conv = nn.ModuleList()
        for l in range(out_levels):
            self.out_conv.append(
                Conv2d(
                    in_channels=hidden_channels[l],
                    out_channels=out_channels[l],
                    normalization=None)
                )

    def forward(self, data):      
        ### Up and Lateral Convolution ###
        prev_features = self.up_conv[0](data)
        lateral_features = []
        lateral_features.append(self.lateral_conv[0](prev_features))
        for l in range(1, self.levels):
            prev_features = self.up_conv[l](prev_features)
            lateral_features.append(self.lateral_conv[l](prev_features))        

        ### Down Convolution ###
        down_features = [None]*(self.levels)
        down_features[self.levels-1] = lateral_features[self.levels-1]
        for l in range(self.levels-1,0,-1):
            down_features[l-1] = self.down_conv[l-1](F.interpolate(down_features[l], scale_factor=2, mode="bilinear") + lateral_features[l-1])

        ### Out Convolution ###
        out_features = []
        for l in range(self.out_levels):
            out_features.append(self.out_conv[l](down_features[l]))

        if self.low_res_first:
            out_features = out_features[::-1]
        
        return out_features

class FPN_v2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 base_channels=8,
                 ):
        """
        Args:
            in_channels: The expected channel size for the input data.
            out_channels: The channel size for the output features.
            base_channels: The base hidden channel size inside the encoder.
        """
        super(FPN_v2, self).__init__()

        self.conv0 = nn.Sequential(
            Conv2d(in_channels=in_channels, out_channels=base_channels, kernel_size=3, padding=1),
            Conv2d(in_channels=base_channels, out_channels=base_channels, kernel_size=3, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2d(in_channels=base_channels, out_channels=base_channels*2, kernel_size=5, padding=2, stride=2),
            Conv2d(in_channels=base_channels*2, out_channels=base_channels*2, kernel_size=3, padding=1),
            Conv2d(in_channels=base_channels*2, out_channels=base_channels*2, kernel_size=3, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(in_channels=base_channels*2, out_channels=base_channels*4, kernel_size=5, padding=2, stride=2),
            Conv2d(in_channels=base_channels*4, out_channels=base_channels*4, kernel_size=3, padding=1),
            Conv2d(in_channels=base_channels*4, out_channels=base_channels*4, kernel_size=3, padding=1),
        )

        self.conv3 = nn.Sequential(
            Conv2d(in_channels=base_channels*4, out_channels=base_channels*8, kernel_size=5, padding=2, stride=2),
            Conv2d(in_channels=base_channels*8, out_channels=base_channels*8, kernel_size=3, padding=1),
            Conv2d(in_channels=base_channels*8, out_channels=base_channels*8, kernel_size=3, padding=1),
        )

        self.conv_out = nn.ModuleList()
        self.conv_out.append(Conv2d(base_channels * 2, out_channels[0], normalization=None, nonlinearity=None, bias=False))
        self.conv_out.append(Conv2d(base_channels * 4, out_channels[1], normalization=None, nonlinearity=None, bias=False))
        self.conv_out.append(Conv2d(base_channels * 8, out_channels[2], normalization=None, nonlinearity=None, bias=False))
        self.conv_out.append(Conv2d(base_channels * 8, out_channels[3], normalization=None, nonlinearity=None, bias=False))

        self.conv_inner = nn.ModuleList()
        self.conv_inner.append(Conv2d(base_channels, base_channels * 2, normalization=None, nonlinearity=None, bias=True))
        self.conv_inner.append(Conv2d(base_channels * 2, base_channels * 4, normalization=None, nonlinearity=None, bias=True))
        self.conv_inner.append(Conv2d(base_channels * 4, base_channels * 8, normalization=None, nonlinearity=None, bias=True))

    def forward(self, tensor):
        conv0 = self.conv0(tensor)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        output_features_3 = self.conv_out[3](conv3)
        intra_feat = F.interpolate(conv3, scale_factor=2, mode="bilinear") + self.conv_inner[2](conv2)
        output_features_2 = self.conv_out[2](intra_feat)
        intra_feat = F.interpolate(conv2, scale_factor=2, mode="bilinear") + self.conv_inner[1](conv1)
        output_features_1 = self.conv_out[1](intra_feat)
        intra_feat = F.interpolate(conv1, scale_factor=2, mode="bilinear") + self.conv_inner[0](conv0)
        output_features_0 = self.conv_out[0](intra_feat)

        return (output_features_3, output_features_2, output_features_1, output_features_0)

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
        self.conv8 = Conv3d(c*4, c*4, kernel_size=(1,3,3), padding=(0,1,1), normalization=None, nonlinearity=None)

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

        self.deconv7 = ConvTranspose3d(c * 4, c * 4, kernel_size=(1,3,3), stride=2, padding=(0,1,1), output_padding=1)
        self.deconv8 = ConvTranspose3d((c*4) + (c*2), c * 4, kernel_size=(1,3,3), stride=2, padding=(0,1,1), output_padding=1)
        self.conv9 = Conv3d((c*4)+c, c * 4, kernel_size=(1,3,3), padding=(0,1,1))
        self.conv10 = Conv3d((c*4)+in_channels, c * 4, kernel_size=(1,3,3), padding=(0,1,1), normalization=None, nonlinearity=None)

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

