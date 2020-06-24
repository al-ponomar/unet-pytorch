import torch
import torchvision
import torchvision.transforms as standard_transforms
from src.model import UNet

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

testset = torchvision.datasets.VOCSegmentation(root='/Users/stranger/PycharmProjects/unet-pytorch/voc_data', year='2008', image_set='test',
                                     download=True, transform=input_transform, target_transform=target_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

net = UNet()
net.to(device)
net.load_state_dict('path')

correct = 0
total = 0
with torch.no_grad():
    for i, data in enumerate(testloader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy %d procent', correct / total * 100)