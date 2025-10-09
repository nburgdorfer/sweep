import torch.nn as nn
import torch.nn.functional as F

from src.components.layers import Conv2d

# from torchvision.ops import DeformConv2d


class BasicEncoder(nn.Module):
    def __init__(self, in_channels, c, out_channels, decode=False):
        super(BasicEncoder, self).__init__()
        self.decode = decode

        # H x W
        conv0 = [
            Conv2d(in_channels=in_channels, out_channels=c, kernel_size=3, padding=1)
        ]
        conv0.append(Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1))
        self.conv0 = nn.Sequential(*nn.ModuleList(conv0))

        # H/2 x W/2
        conv1 = [
            Conv2d(
                in_channels=c, out_channels=c * 2, kernel_size=5, stride=2, padding=2
            )
        ]
        conv1.append(
            Conv2d(in_channels=c * 2, out_channels=c * 2, kernel_size=3, padding=1)
        )
        conv1.append(
            Conv2d(in_channels=c * 2, out_channels=c * 2, kernel_size=3, padding=1)
        )
        self.conv1 = nn.Sequential(*nn.ModuleList(conv1))

        # H/4 x W/4
        conv2 = [
            Conv2d(
                in_channels=c * 2,
                out_channels=c * 4,
                kernel_size=5,
                stride=2,
                padding=2,
            )
        ]
        conv2.append(
            Conv2d(in_channels=c * 4, out_channels=c * 4, kernel_size=3, padding=1)
        )
        conv2.append(
            Conv2d(in_channels=c * 4, out_channels=c * 4, kernel_size=3, padding=1)
        )
        self.conv2 = nn.Sequential(*nn.ModuleList(conv2))

        if self.decode:
            self.conv3 = Conv2d(
                in_channels=c * 2, out_channels=c * 4, kernel_size=3, padding=1
            )
            self.conv4 = Conv2d(
                in_channels=c, out_channels=c * 2, kernel_size=3, padding=1
            )
            self.out1 = Conv2d(
                in_channels=c * 4, out_channels=c * 2, kernel_size=1, padding=0
            )
            self.out2 = Conv2d(
                in_channels=c * 2,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
                normalization=None,
                nonlinearity=None,
            )
        else:
            self.out1 = Conv2d(
                in_channels=c * 4,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
                normalization=None,
                nonlinearity=None,
            )

    def forward(self, img):
        z0 = self.conv0(img)  # [c x H x W]
        z1 = self.conv1(z0)  # [c*2 x H/2 x W/2]
        z2 = self.conv2(z1)  # [c*4 x H/4 x W/4]

        if self.decode:
            f1 = self.out1(
                F.interpolate(z2, scale_factor=2, mode="bilinear") + self.conv3(z1)
            )  # [c*2 x H/2 x W/2]
            out = self.out2(
                F.interpolate(f1, scale_factor=2, mode="bilinear") + self.conv4(z0)
            )  # [out_channels x H x W]
        else:
            out = self.out1(z2)  # [out_channels x H x W]

        return out


class FPN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        base_channels=8,
        levels=4,
        block_size=3,
    ):
        """
        Args:
            in_channels: The expected channel size for the input data.
            out_channels: The channel size for the output features.
            base_channels: The base hidden channel size inside the encoder.
            levels: The number of total pyramid layers.
            block_size: The number of convolution layers per convolution block.
        """
        super(FPN, self).__init__()

        assert levels > 1
        assert block_size > 0
        assert base_channels > 0

        self.levels = levels

        # build hidden channels list
        hidden_channels = []
        for l in range(self.levels):
            hidden_channels.append(base_channels * (2**l))

        ### Input Convolution Layers
        self.input_conv = nn.Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels[0],
                kernel_size=3,
                padding=1,
            ),
            Conv2d(
                in_channels=hidden_channels[0],
                out_channels=hidden_channels[0],
                kernel_size=3,
                padding=1,
            ),
        )

        ### Up Convolution Layers
        self.up_conv = nn.ModuleList()
        for i in range(1, levels):
            up_conv = [
                Conv2d(
                    in_channels=hidden_channels[i - 1],
                    out_channels=hidden_channels[i],
                    kernel_size=5,
                    padding=2,
                    stride=2,
                )
            ]
            for _ in range(block_size - 1):
                up_conv.append(
                    Conv2d(
                        in_channels=hidden_channels[i],
                        out_channels=hidden_channels[i],
                        kernel_size=3,
                        padding=1,
                    )
                )
            self.up_conv.append(nn.Sequential(*nn.ModuleList(up_conv)))

        ### Lateral Convolution Layers
        self.lateral_conv = nn.ModuleList()
        for i in range(1, levels):
            self.lateral_conv.append(
                Conv2d(
                    hidden_channels[i - 1],
                    hidden_channels[i],
                    normalization=None,
                    nonlinearity=None,
                    bias=True,
                )
            )

        ### Output Convolution Layers
        self.output_conv = nn.ModuleList()
        for i in range(1, levels):
            self.output_conv.append(
                Conv2d(
                    hidden_channels[i],
                    out_channels[i - 1],
                    normalization=None,
                    nonlinearity=None,
                    bias=False,
                )
            )
        self.output_conv.append(
            Conv2d(
                hidden_channels[levels - 1],
                out_channels[levels - 1],
                normalization=None,
                nonlinearity=None,
                bias=False,
            )
        )

        # ### Output Convolution Layers
        # self.output_conv = nn.ModuleList()
        # for i in range(1, levels):
        #     self.output_conv.append(
        #         DeformConv2d(
        #             hidden_channels[i],
        #             out_channels[i - 1],
        #             kernel_size=3,
        #             stride=1,
        #             padding=1,
        #         )
        #     )
        # self.output_conv.append(
        #     DeformConv2d(
        #         hidden_channels[levels - 1],
        #         out_channels[levels - 1],
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #     )
        # )

    def forward(self, tensor, resolution_level):
        conv0 = self.input_conv(tensor)

        # Up Convolution
        convs = [conv0]
        for i in range(self.levels - 1):
            convs.append(self.up_conv[i](convs[i]))

        # Lateral Convolution
        # output_features = []
        lateral_features = convs[3]
        for i in range(self.levels - 1, 0, -1):
            output_features = self.output_conv[i](lateral_features)

            # skip computing higher resolution features if they are not needed
            if i == resolution_level:
                return output_features

            lateral_features = F.interpolate(
                convs[i], scale_factor=2, mode="bilinear"
            ) + self.lateral_conv[i - 1](convs[i - 1])

        return self.output_conv[0](lateral_features)
