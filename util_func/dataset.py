from torchvision import transforms, datasets
from typing import *
import torch
import os
from torch.utils.data import Dataset


# list of all datasets
DATASETS = ["imagenet", "cifar10", "cifar100", "mnist", "fashion_mnist", "tiny_imagenet"]


def get_dataset(config, split: str) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    if config.dataset == "imagenet":
        return _imagenet(config, split)
    elif config.dataset == "cifar10":
        return _cifar10(config, split)
    elif config.dataset == "cifar100":
        return _cifar100(config, split)
    elif config.dataset == "mnist":
        return _mnist10(config, split)
    elif config.dataset == "tiny_imagenet":
        return _tinyimagenet(config, split)
    elif config.dataset == "fashion_mnist":
        return _fashion_mnist10(config, split)
    

def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    if dataset == "imagenet":
        return 1000
    elif dataset == "cifar10":
        return 10
    elif dataset == "cifar100":
        return 100
    elif dataset == "mnist":
        return 10
    elif dataset == "fashion_mnist":
        return 10

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STDDEV = (0.2023, 0.1994, 0.2010)

_MNIST_MEAN = (0.1307,)
_MNIST_STDDEV = (0.3081,)

_FASHION_MNIST_MEAN = (0.28604,)
_FASHION_MNIST_STDDEV = (0.35302,)

_TINY_MEAN = [0.480, 0.448, 0.398]
_TINY_STD = [0.277, 0.269, 0.282]

def _mnist10(config, split: str) -> Dataset:
    if split == "train":
        return datasets.MNIST(config.data_path, train=True, download=True, transform=transforms.Compose([
#             transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_MNIST_MEAN, _MNIST_STDDEV)
        ]))
    
    elif split == "test":
        return datasets.MNIST(config.data_path, train=False, download=True, transform=transforms.Compose([
#             transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_MNIST_MEAN, _MNIST_STDDEV)
        ]))
    

def _fashion_mnist10(config, split: str) -> Dataset:
    if split == "train":
        return datasets.FashionMNIST(config.data_path, train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_FASHION_MNIST_MEAN, _FASHION_MNIST_STDDEV)
        ]))
    elif split == "test":
        return datasets.FashionMNIST(config.data_path, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_FASHION_MNIST_MEAN, _FASHION_MNIST_STDDEV)
        ]))

def _cifar10(config, split: str) -> Dataset:
    if split == "train":
        return datasets.CIFAR10(config.data_path, train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STDDEV)
            
        ]))
    elif split == "test":
        return datasets.CIFAR10(config.data_path, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STDDEV)
        ]))
    
def _cifar100(config, split: str) -> Dataset:
    if split == "train":
        return datasets.CIFAR100(config.data_path, train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            
        ]))
    elif split == "test":
        return datasets.CIFAR100(config.data_path, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])) 
    


def _imagenet(config, split: str) -> Dataset:
    if split == "train":
        subdir = os.path.join(config.data_path, "train")
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STDDEV)
        ])
    elif split == "test":
        subdir = os.path.join(config.data_path, "val")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STDDEV)
        ])
    return datasets.ImageFolder(subdir, transform)

def _tinyimagenet(config, split: str) -> Dataset:
    if split == "train":
        subdir = os.path.join(config.data_path, "train")
        transform = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_TINY_MEAN, _TINY_STD)
        ])
    elif split == "test":
        subdir = os.path.join(config.data_path, "val")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_TINY_MEAN, _TINY_STD)
        ])
    return datasets.ImageFolder(subdir, transform)