import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms as standard_transforms

import matplotlib.pyplot as plt
import numpy as np
from src.model import UNet
import torch.optim as optim

# net = UNet()
# test_batch = np.ones((5, 3, 256, 256))
# test_gt = np.ones((5, 1, 256, 256))
# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# optimizer.zero_grad()
# images = torch.from_numpy(test_batch)
# outputs = net(images)
# labels = torch.from_numpy(test_gt)
# # labels = labels.squeeze(1)
# # labels = labels.long()
# # print(labels.size(), labels.type())
# # print(outputs.size(), outputs.type())
#
# loss = criterion(outputs, labels)

# print(outputs.size())
# output_img = outputs.data.numpy()
# print(output_img.shape)
#
# plt.imshow(output_img[0,0,:,:])
# plt.show()
# torch.Size([1, 3, 280, 280]) torch.Size([1, 1, 280, 280])



# def imshow(img, gt=False):
#     npimg = img.numpy()
#     if not gt:
#         plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     else:
#         plt.imshow(npimg[0, :, :])
#
#     plt.show()
#
# mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

input_transform = standard_transforms.Compose([
    standard_transforms.CenterCrop(256),
    standard_transforms.ToTensor()
])
target_transform = standard_transforms.Compose([
    standard_transforms.CenterCrop(256),
    standard_transforms.ToTensor()
])
trainset = torchvision.datasets.VOCSegmentation(root='/Users/stranger/PycharmProjects/unet-pytorch/voc_data', year='2012', image_set='train',
                                         download=True, transform=input_transform,  target_transform=target_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

print('training data loading finished')
print(len(trainloader))

# testset = torchvision.datasets.VOCSegmentation(root='/Users/stranger/PycharmProjects/unet-pytorch/voc_data', year='2012', image_set='trainval',
#                                      download=True, transform=input_transform, target_transform=target_transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

# print(len(testloader))

print('validation data loading finished')

# dataiter = iter(testloader)
# images, labels = dataiter.next()
# print(images.size(), labels.size())
# print(images.numpy().shape, labels.numpy().shape)
# imshow(images[0])

# net = UNet()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# print('Started training')

net = UNet()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        running_loss += loss.item()
        if i % 200 == 199:
            print('[%d, %d] loss %.3f' % (epoch + 1, i + 1, running_loss / 200))
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

