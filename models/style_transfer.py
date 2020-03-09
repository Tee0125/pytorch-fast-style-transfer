import torch
from torch import nn


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, has_relu=True):
        super().__init__()

        self.net = nn.Sequential(nn.ReflectionPad2d(kernel_size//2),
                                 nn.Conv2d(in_channels, out_channels, kernel_size, stride),
                                 nn.InstanceNorm2d(out_channels, affine=True))

        self.relu = nn.ReLU() if has_relu else None

    def forward(self, x):
        x = self.net(x)

        if self.relu:
            x = self.relu(x)

        return x


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        i = in_channels
        o = out_channels

        self.net = nn.Sequential(nn.Conv2d(i, o, 3, 1, 1),
                                 nn.InstanceNorm2d(o, affine=True),
                                 nn.ReLU(),
                                 nn.Conv2d(o, o, 3, 1, 1),
                                 nn.InstanceNorm2d(o, affine=True))

    def forward(self, x):
        return self.net(x) + x


class Deconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, has_relu=True):
        super().__init__()

        self.net = nn.Sequential(nn.Upsample(scale_factor=scale_factor),
                                 nn.ReflectionPad2d(kernel_size // 2),
                                 nn.Conv2d(in_channels, out_channels, kernel_size),
                                 nn.InstanceNorm2d(out_channels, affine=True))

        self.relu = nn.ReLU() if has_relu else None

    def forward(self, x):
        x = self.net(x)

        if self.relu:
            x = self.relu(x)

        return x


# https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf
class StyleTransfer(nn.Module):
    name = 'style_transfer'

    def __init__(self):
        super().__init__()

        # to fit output in range(0, 1) Sigmoid is used instead of Tanh
        self.net = nn.Sequential(Conv2d(3, 32, 9, 1),
                                 Conv2d(32, 64, 3, 2),
                                 Conv2d(64, 128, 3, 2),

                                 Residual(128, 128),
                                 Residual(128, 128),
                                 Residual(128, 128),
                                 Residual(128, 128),
                                 Residual(128, 128),

                                 Deconv2d(128, 64, 3, 2.),
                                 Deconv2d(64, 32, 3, 2.),
                                 Conv2d(32, 3, 9, 1, False),

                                 nn.Sigmoid())

    def forward(self, x):
        return self.net(x)

