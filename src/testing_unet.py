import torch
import torchvision
import torchvision.transforms as standard_transforms
from src.model import UNet
import numpy as np
import matplotlib.pyplot as plt


def imshow(npimg):
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

input_transform = standard_transforms.Compose([
    standard_transforms.CenterCrop(256),
    standard_transforms.ToTensor()
])
target_transform = standard_transforms.Compose([
    standard_transforms.CenterCrop(256),
    standard_transforms.ToTensor()
])

testset = torchvision.datasets.VOCSegmentation(root='/home/elena/PycharmProjects/unet-pytorch/voc_data', year='2008',
                                               image_set='val',
                                               download=True, transform=input_transform,
                                               target_transform=target_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=8)

net = UNet()
net.to(device)
# net.load_state_dict(
#     torch.load('/home/elena/PycharmProjects/unet-pytorch/src/wandb/run-20200624_081139-3rg4aznj/unet_wb.pt'))

right_pixels = 0
count = 0
with torch.no_grad():
    for i, data in enumerate(testloader):
        images, labels = data
        labels[labels != 0] = 1.0
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        outputs[outputs > 0.5] = 1
        outputs[outputs < 0.5] = 0

        # output = outputs.to('cpu')[0].data.numpy()
        # label = labels.to('cpu')[0].data.numpy()
        # image = images.to('cpu')[0].data.numpy()
        # imshow(image)
        # imshow(output)
        # imshow(label)

        # labels = torch.squeeze(labels, dim=1)
        right_pixels_current = (outputs == labels).sum().item() / (256 * 256) * 100
        right_pixels += right_pixels_current
        count += labels.size(0)
        # print(right_pixels_current)
print('Accuracy %d procent' % (right_pixels / count))
