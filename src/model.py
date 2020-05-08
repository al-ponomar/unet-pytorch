import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.module):

    def down_block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, 3, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
    def up_block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, 2),
            nn.Conv2d(out_dim, out_dim, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 3),
            nn.ReLU(inplace=True)
        )

    def __init__(self):
        super(UNet, self).__init__()
        self.down1 = self.down_block(3, 32)
        self.down2 = self.down_block(32, 64)
        self.down3 = self.down_block(64, 128)
        self.down4 = self.down_block(128, 256)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(inplace=True)
        )
        self.up1 = self.up_block(512, 256)
        self.up2 = self.up_block(256, 128)
        self.up3 = self.up_block(128, 64)
        self.up3 = self.up_block(64, 32)

        self.output = nn.Sequential(
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )


    # def forward(self, x):




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