import torch
import matplotlib.pyplot as plt
import numpy as np


def get_accuracy(testloader, net, device):
    right_pixels = 0
    count = 0
    with torch.no_grad():
        for i, data in enumerate(testloader):
            test_images, test_labels = data
            test_labels[test_labels != 0] = 1.0
            test_images, test_labels = test_images.to(device), test_labels.to(device)
            outputs = net(test_images)
            outputs[outputs > 0.5] = 1
            outputs[outputs < 0.5] = 0

            # output = outputs.to('cpu')[0].data.numpy()
            # label = test_labels.to('cpu')[0].data.numpy()
            # image = test_images.to('cpu')[0].data.numpy()
            # imshow(image)
            # imshow(output)
            # imshow(label)

            # labels = torch.squeeze(labels, dim=1)
            right_pixels_current = (outputs == test_labels).sum().item() / (256 * 256) * 100
            right_pixels += right_pixels_current
            count += test_labels.size(0)
            # print(right_pixels_current)
        return right_pixels / count


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

#
# net = UNet()
# net.to(device)
# net.load_state_dict(
#     torch.load('/home/elena/PycharmProjects/unet-pytorch/src/wandb/run-20200624_081139-3rg4aznj/unet_wb.pt'))
#
