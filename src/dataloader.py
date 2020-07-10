import torchvision
import torchvision.transforms as standard_transforms
import torch

def load_voc_data(root, year):
    input_transform = standard_transforms.Compose([
        standard_transforms.CenterCrop(256),
        standard_transforms.ToTensor()
    ])
    target_transform = standard_transforms.Compose([
        standard_transforms.CenterCrop(256),
        standard_transforms.ToTensor()
    ])
    trainset = torchvision.datasets.VOCSegmentation(root=root,
                                                    year=year,
                                                    image_set='train',
                                                    download=True, transform=input_transform,
                                                    target_transform=target_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=8)

    print('training data loading finished')

    testset = torchvision.datasets.VOCSegmentation(root=root,
                                                   year=year,
                                                   image_set='val',
                                                   download=True, transform=input_transform,
                                                   target_transform=target_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=8)

    print('validation data loading finished')

    return trainloader, testloader