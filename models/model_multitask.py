# U-Net implementation is available from
# https://github.com/milesial/Pytorch-UNet
# under GNU General Public License v3.0
#
# Copyright (C) 2017 https://github.com/milesial

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x, w):
        return x


class MulticlassClassification(nn.Module):
    def __init__(self, in_channels, out_channels, num_class):
        super(MulticlassClassification, self).__init__()
        self.out_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.fc = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(120 * 120, num_class))

    def forward(self, x):
        x1 = self.out_conv(x)
        x2 = x1.view(-1, 120 * 120)
        x_out = self.fc(x2)
        return x_out


class ConvRegression(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvRegression, self).__init__()
        self.out_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.fc = nn.Sequential(
            nn.Linear(120 * 120 + 4, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.ReLU(inplace=True),)

    def forward(self, x, w):
        x1 = self.out_conv(x)
        x1 = x1.view(-1, 120 * 120)
        x2 = w.view(-1, 4)
        x3 = torch.cat((x1, x2), dim=1)
        x_out = self.fc(x3)
        return x_out


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1))

    def forward(self, x):
        x_out = self.conv(x)
        return x_out


class MultiTaskModel(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(MultiTaskModel, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.outc = OutConv(64, n_classes)
        self.outr = ConvRegression(64, 1)
        self.outb = MulticlassClassification(64, 1, 4)

    def forward(self, x, w):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x_out_c = self.outc(x)
        x_out_r = self.outr(x, w)
        x_out_b = self.outb(x)

        return x_out_c, x_out_r, x_out_b
