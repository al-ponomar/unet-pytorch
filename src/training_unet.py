import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as standard_transforms
import os
import matplotlib.pyplot as plt
import numpy as np
from model import UNet
from testing_unet import get_accuracy
import torch.optim as optim
import wandb
import argparse

hyperparameter_defaults = dict(
    learning_rate=0.001,
    epochs=10
)

wandb.init(project="test")


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


def train():
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=8)

    print('training data loading finished')

    testset = torchvision.datasets.VOCSegmentation(root='/home/elena/PycharmProjects/unet-pytorch/voc_data', year='2008',
                                                   image_set='val',
                                                   download=True, transform=input_transform,
                                                   target_transform=target_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=8)

    print('validation data loading finished')



    wandb.init(config=hyperparameter_defaults)
    config = wandb.config

    net = UNet()
    criterion = nn.BCEWithLogitsLoss()
    if config.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=config.learning_rate)
    else:
        optimizer = optim.SGD(net.parameters(), lr=config.learning_rate)
    net.to(device)

    wandb.watch(net)

    for epoch in range(config.epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            images, labels = data
            labels = labels.data
            labels[labels != 0] = 1.0
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # wandb.log({"training loss:", loss.item()})
            wandb.log({"Batch Loss": loss.item()})
        if epoch % 4 == 0:
            accuracy = get_accuracy(testloader, net, device)
            wandb.log({"Accuracy": accuracy})
            print("Accuracy:", accuracy)

        print('%d loss %.3f' % (epoch + 1, running_loss / len(trainloader)))
        wandb.log({"total_loss": running_loss / len(trainloader), "Epoch": epoch + 1})

    print('Finished training')

    # path_to_model = '/home/elena/PycharmProjects/unet-pytorch/models/unet2.pt'
    torch.save(net.state_dict(), os.path.join(wandb.run.dir, 'unet_wb.pt'))

if __name__ == "__main__":
    train()