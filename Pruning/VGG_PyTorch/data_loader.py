import os

import torch

import torchvision.transforms as transforms
import torchvision.datasets as datasets

def data_loader(dataset, batch_size=256, workers=1, pin_memory=True):
    if dataset=='cifar10':
        # Define the transforms to apply to the data
        transform = transforms.Compose([
            transforms.ToTensor(),      # Convert the image to a PyTorch tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the pixel values to have zero mean and unit variance
        ])

        # Load the CIFAR-10 dataset
        train_dataset = datasets.CIFAR10(root='./data/{}/'.format(dataset), train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data/{}/'.format(dataset), train=False, download=True, transform=transform)

        # Create data loaders to load the data in batches
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=pin_memory)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory)

        return train_loader, test_loader
    
    elif dataset=='imagenet':
        traindir = os.path.join('/home/cguo/imagenet-raw-data/', 'train')
        valdir = os.path.join('/home/cguo/imagenet-raw-data/', 'val')
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        )
        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=pin_memory,
            sampler=None
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=pin_memory
        )
    return train_loader, val_loader
