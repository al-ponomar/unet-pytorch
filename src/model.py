import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim

class Unet(nn.Module):

    def __init__(self):
        super(Unet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)

        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for dim in size:
            num_features *= dim
        return num_features

net = Unet()
optimizer = optim.SGD(net.parameters(), lr=0.001)
optimizer.zero_grad()

print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())

input = torch.randn(1, 32, 32)
input = input.unsqueeze(0)
out = net(input)
target = torch.randn(10)
print(target.size())
target = target.view(1, -1)
print(target.size())

criterion = nn.MSELoss()
loss = criterion(out, target)


net.zero_grad()
print(net.conv1.bias.grad)
loss.backward()
print(net.conv1.bias.grad)
optimizer.step()