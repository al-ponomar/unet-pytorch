import torch.nn as nn
import torch.nn.functional as F
import torch


class UNet(nn.Module):

    def _block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, padding=2),
            nn.BatchNorm2d(num_features=out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 3),
            nn.BatchNorm2d(num_features=out_dim),
            nn.ReLU(inplace=True)
        )


    def __init__(self):
        super(UNet, self).__init__()
        self.down1 = self._block(3, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down2 = self._block(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down3 = self._block(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down4 = self._block(128, 256)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(inplace=True)
        )
        self.upconv4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up4 = self._block(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up3 = self._block(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up2 = self._block(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.up1 = self._block(64, 32)

        self.output = nn.Sequential(
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.float()

        down1 = self.down1(x)
        down2 = self.down2(self.pool1(down1))
        down3 = self.down3(self.pool2(down2))
        down4 = self.down4(self.pool3(down3))

        bottleneck = self.bottleneck(self.pool4(down4))

        up4 = self.upconv4(bottleneck)
        up4 = torch.cat((down4, up4), dim=1)
        up4 = self.up4(up4)

        up3 = self.upconv3(up4)
        up3 = torch.cat((down3, up3), dim=1)
        up3 = self.up3(up3)

        up2 = self.upconv2(up3)
        up2 = torch.cat((down2, up2), dim=1)
        up2 = self.up2(up2)

        up1 = self.upconv1(up2)
        up1 = torch.cat((down1, up1), dim=1)
        up1 = self.up1(up1)

        out = self.output(up1)
        return out


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
