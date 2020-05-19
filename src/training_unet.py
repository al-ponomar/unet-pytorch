import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms as standard_transforms

import matplotlib.pyplot as plt
import numpy as np
# from src.model import UNet
import torch.optim as optim

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

input_transform = standard_transforms.Compose([
    standard_transforms.CenterCrop(280),
    standard_transforms.ToTensor()
])
target_transform = standard_transforms.Compose([
    standard_transforms.CenterCrop(280),
    standard_transforms.ToTensor()
])
# trainset = torchvision.datasets.VOCSegmentation(root='/Users/stranger/PycharmProjects/unet-pytorch/voc_data', year='2012', image_set='train',
#                                          download=True, transform=input_transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
#
# print('training data loading finished')

testset = torchvision.datasets.VOCSegmentation(root='/Users/stranger/PycharmProjects/unet-pytorch/voc_data', year='2012', image_set='trainval',
                                     download=True, transform=input_transform, target_transform=target_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

print(len(testloader))

print('validation data loading finished')

# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

for vi, data in enumerate(testloader, 0):
    inputs, gts = data
    imshow(inputs[0])
    imshow(gts[0])
    break

# dataiter = iter(testloader)
# images, labels = dataiter.next()
# imshow(images[0])

# net = UNet()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# print('Started training')
# for epoch in range(2):
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         images, labels = data
#
#         optimizer.zero_grad()
#         outputs = net(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#         if i % 200 == 199:
#             print('[%d, %d] loss %.3f' % (epoch + 1, i + 1, running_loss / 200))
#             running_loss = 0
# print('Finished training')
#
# path_to_model = '/Users/stranger/PycharmProjects/unet-pytorch/models/' + 'test2.pt'
# torch.save(net.state_dict(), path_to_model)
#
#

