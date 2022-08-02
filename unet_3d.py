import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=1, squeeze=False):
        super(UNet, self).__init__()
        self.conv1 = Conv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 64)
        self.out = OutConv(64, 1)
        self.squeeze = squeeze


    def forward(self, x):
        if self.squeeze:
            x = x.unsqueeze(1)
        x1 = self.conv1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        if self.squeeze:
            x = x.squeeze(1)
        return x


class OutConv(nn.Module):
    def __init__(self, in_size, out_size):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size=3, padding=1))

    def forward(self, x):
        x = self.conv(x)
        return x


class Conv(nn.Module):
    def __init__(self, in_size, out_size):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_size),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_size, out_size, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_size, out_size):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool3d(2),
            Conv(in_size, out_size)
        )

    def forward(self, x):
        x = self.down(x)
        return x


class Up(nn.Module):
    def __init__(self, in_size, out_size):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose3d(in_size, in_size, kernel_size=2, stride=2)
        self.conv = Conv(in_size * 2, out_size)

    def forward(self, x1, x2):
        up = self.up(x1)
        out = torch.cat([up, x2], dim=1)
        out = self.conv(out)
        return out
