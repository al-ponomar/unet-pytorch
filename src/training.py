import torch
import torch.nn as nn
import os
from src.model import UNet
from src.evaluation import get_accuracy, imshow
from src.dataloader import load_voc_data
import wandb

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

hyperparameter_defaults = dict(
    learning_rate=0.001,
    epochs=5,
    optimizer='adam'
)


def train():
    trainloader, testloader = load_voc_data(root='/home/elena/PycharmProjects/unet-pytorch/voc_data',
                                            year='2008')


    wandb.init(project="test",
               config=hyperparameter_defaults)

    net = UNet()

    criterion = nn.BCEWithLogitsLoss()
    if wandb.config.optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=wandb.config.learning_rate)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=wandb.config.learning_rate)
    net.to(device)

    wandb.watch(net)

    for epoch in range(wandb.config.epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            images, labels = data
            labels = labels.data
            labels[labels != 0] = 1.0
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            if i == 0:
                wandb.log({"masked_examples": [wandb.Image(j) for j in images]})
                wandb.log({"labels": [wandb.Image(j) for j in labels]})
                wandb.log({"predictions": [wandb.Image(j) for j in outputs]})
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if epoch % 1 == 0:
            accuracy = get_accuracy(testloader, net, device)
            wandb.log({"Accuracy": accuracy})
            print("Accuracy:", accuracy)

        print('%d loss %.3f' % (epoch + 1, running_loss / len(trainloader)))
        wandb.log({"total_loss": running_loss / len(trainloader), "Epoch": epoch + 1})

    print('Finished training')

    torch.save(net.state_dict(), os.path.join(wandb.run.dir, 'unet_wb.pt'))


if __name__ == "__main__":
    train()
