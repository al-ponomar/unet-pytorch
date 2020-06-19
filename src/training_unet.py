import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms as standard_transforms

import matplotlib.pyplot as plt
import numpy as np
from src.model import UNet
import torch.optim as optim


def imshow(img):
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    if npimg.shape[2] == 1:
        npimg = npimg[:, :, 0]
        zers = np.zeros((256, 256, 3))
        zers[:, :, 0] = npimg
        zers[:, :, 1] = npimg
        zers[:, :, 2] = npimg
        npimg = zers
    plt.imshow(npimg)
    plt.show()


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

# net = UNet()
# test_batch = np.ones((5, 3, 256, 256))
# test_gt = np.ones((5, 1, 256, 256))
# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# optimizer.zero_grad()
# images = torch.from_numpy(test_batch)
# outputs = net(images)

input_transform = standard_transforms.Compose([
    standard_transforms.CenterCrop(256),
    standard_transforms.ToTensor()
])
target_transform = standard_transforms.Compose([
    standard_transforms.CenterCrop(256),
    standard_transforms.ToTensor()
])
trainset = torchvision.datasets.VOCSegmentation(root='/home/elena/PycharmProjects/unet-pytorch/voc_data', year='2008',
                                                image_set='train',
                                                download=True, transform=input_transform,
                                                target_transform=target_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

print('training data loading finished')
print(len(trainloader))

# testset = torchvision.datasets.VOCSegmentation(root='/Users/stranger/PycharmProjects/unet-pytorch/voc_data', year='2012', image_set='trainval',
#                                      download=True, transform=input_transform, target_transform=target_transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

# print(len(testloader))

print('validation data loading finished')

net = UNet()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)
net.to(device)
for epoch in range(100):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        images, labels = data
        labels = labels.data
        labels[labels!=0] = 1.0
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        if i < 1:
            output = outputs.to('cpu')[0].data
            label = labels.to('cpu')[0].data
            imshow(output)
            imshow(label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print('%d loss %.3f' % (epoch + 1, running_loss / len(trainloader)))

print('Finished training')
#
# path_to_model = '/home/elena/PycharmProjects/unet-pytorch/models/unet.pt'
# torch.save(net.state_dict(), path_to_model)
