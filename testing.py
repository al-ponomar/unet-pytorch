import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from src.model import Net
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = torchvision.datasets.CIFAR10(root='/Users/stranger/PycharmProjects/unet-pytorch/data', train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

dataiter = iter(testloader)
images, labels = dataiter.next()
# imshow(torchvision.utils.make_grid(images))
print('Ground truth: ', ' '.join(classes[labels[i]] for i in range(4)))

model = Net()
model.to(device)
model.load_state_dict(torch.load('/Users/stranger/PycharmProjects/unet-pytorch/models/test2.pt'))
# model.eval()
outputs = model(images)
#
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join(classes[predicted[i]] for i in range(4)))

correct = 0
total = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels)
        for j in range(4):
            label = labels[j]
            class_correct[label] += int(c[j].item())
            class_total[label] += 1
        total += labels.size(0)
        correct += (labels == predicted).sum().item()
print('Accuracy: ', 100 * correct / total)
for i in range(10):
    print('accuracy for class', classes[i], class_correct[i]/class_total[i] *100)




