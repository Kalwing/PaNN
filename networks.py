import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.v = torch.nn.Parameter(torch.tensor([1.0]))
        self.v.requires_grad = False

        self.conv1 = nn.Conv2d(1, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.loss_parameters = [
            self.v,
        ]
        self.net_parameters = [
            self.conv1,
            self.pool,
            self.conv2,
            self.fc1,
            self.fc2,
            self.fc3,
        ]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def switch_to_loss_training(self, boolean):
        for p in self.loss_parameters:
            p.requires_grad = boolean
        for p in self.net_parameters:
            p.requires_grad = not boolean


def conv2DBlock(input_size, output_size):
    return nn.Sequential(
        nn.Conv2d(input_size, output_size, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(output_size, output_size, 3, padding=1),
        nn.ReLU(),
    )


def copy_crop(x0, x1):
    s_final = x1.shape[-2:]
    s = x0.shape[-2:]
    w_min = (s[0] - s_final[0]) // 2
    h_min = (s[1] - s_final[1]) // 2
    x0 = x0[:, :, w_min : w_min + s_final[0], h_min : h_min + s_final[1]]
    return torch.cat((x0, x1), 1)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.convD1 = conv2DBlock(n_channels, 64)
        self.pool = nn.MaxPool2d(2)
        self.convD2 = conv2DBlock(64, 128)
        self.convD3 = conv2DBlock(128, 256)
        self.convD4 = conv2DBlock(256, 512)

        self.bridge = conv2DBlock(512, 1024)
        UpConv = partial(nn.ConvTranspose2d, kernel_size=2, stride=2)
        self.upconv0 = UpConv(1024, 512)

        self.convU1 = conv2DBlock(1024, 512)
        self.upconv1 = UpConv(512, 256)
        self.convU2 = conv2DBlock(512, 256)
        self.upconv2 = UpConv(256, 128)
        self.convU3 = conv2DBlock(256, 128)
        self.upconv3 = UpConv(128, 64)
        self.convU4 = conv2DBlock(128, 64)
        self.outconv = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        # Encoding
        x0 = self.convD1(x)
        x1 = self.convD2(self.pool(x0))
        x2 = self.convD3(self.pool(x1))
        x3 = self.convD4(self.pool(x2))
        x4 = self.bridge(self.pool(x3))

        # Decoding
        x3 = copy_crop(x3, self.upconv0(x4))
        x3 = self.convU1(x3)
        x2 = copy_crop(x2, self.upconv1(x3))
        x2 = self.convU2(x2)
        x1 = copy_crop(x1, self.upconv2(x2))
        x1 = self.convU3(x1)
        x0 = copy_crop(x0, self.upconv3(x1))
        x0 = self.convU4(x0)
        return self.outconv(x0)


class SmallUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(SmallUNet, self).__init__()
        self.convD1 = conv2DBlock(n_channels, 64)
        self.pool = nn.MaxPool2d(2)
        self.convD2 = conv2DBlock(64, 128)

        self.bridge = conv2DBlock(128, 256)
        UpConv = partial(nn.ConvTranspose2d, kernel_size=2, stride=2)
        self.upconv0 = UpConv(256, 128)

        self.convU1 = conv2DBlock(256, 128)
        self.upconv1 = UpConv(128, 64)
        self.convU2 = conv2DBlock(128, 64)
        self.outconv = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        # Encoding
        x0 = self.convD1(x)
        x1 = self.convD2(self.pool(x0))
        x2 = self.bridge(self.pool(x1))

        # Decoding
        x1 = copy_crop(x1, self.upconv0(x2))
        x1 = self.convU1(x1)
        x0 = copy_crop(x0, self.upconv1(x1))
        x0 = self.convU2(x0)
        out = self.outconv(x0)
        return torch.sigmoid(out)
