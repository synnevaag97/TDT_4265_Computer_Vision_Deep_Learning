from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import typing
import numpy as np
import pathlib
import matplotlib.pyplot as plt 
from utils import torch_image_to_numpy, set_seed
np.random.seed(0)

mean = (0.5, 0.5, 0.5)
std = (.25, .25, .25)

def get_data_dir():
    server_dir = pathlib.Path("/work/datasets/cifar10")
    if server_dir.is_dir():
        return str(server_dir)
    return "data/cifar10"


def load_cifar10(batch_size: int, validation_fraction: float = 0.1
                 ) -> typing.List[torch.utils.data.DataLoader]:
    # Note that transform train will apply the same transform for
    # validation!
    set_seed(0)
    transform_train1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform_train2= transforms.Compose([
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    data_train_list = list((datasets.CIFAR10(get_data_dir(),
                                              train=True,
                                              download=True,
                                              transform=transform_train1),datasets.CIFAR10(get_data_dir(),
                                              train=True,
                                              download=True,
                                              transform=transform_train2)))
    """data_train_list = data_train_list.append(datasets.CIFAR10(get_data_dir(),
                                              train=True,
                                              download=True,
                                              transform=transform_train1))
    data_train_list = data_train_list.append(datasets.CIFAR10(get_data_dir(),
                                              train=True,
                                              download=True,
                                              transform=transform_train2))"""
    
    data_train = torch.utils.data.ConcatDataset(data_train_list)
    """data_train = datasets.CIFAR10(get_data_dir(),
                                    train=True,
                                    download=True,
                                    transform=transform_train1)"""
    
    data_test = datasets.CIFAR10(get_data_dir(),
                                 train=False,
                                 download=True,
                                 transform=transform_test)
    
    indices = list(range(len(data_train)))
    split_idx = int(np.floor(validation_fraction * len(data_train)))

    val_indices = np.random.choice(indices, size=split_idx, replace=False)
    train_indices = list(set(indices) - set(val_indices))

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)

    dataloader_train = torch.utils.data.DataLoader(data_train,
                                                   sampler=train_sampler,
                                                   batch_size=batch_size,
                                                   num_workers=2,
                                                   drop_last=True)

    dataloader_val = torch.utils.data.DataLoader(data_train,
                                                 sampler=validation_sampler,
                                                 batch_size=batch_size,
                                                 num_workers=2)

    dataloader_test = torch.utils.data.DataLoader(data_test,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=2)

    return dataloader_train, dataloader_val, dataloader_test
