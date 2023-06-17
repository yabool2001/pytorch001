import torch
from torch import nn
import matplotlib as plt
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import timeit

# Setup training data
train_data = datasets.FashionMNIST(
    root="", # where to download data to?
    train=True, # get training data
    download=True, # download data if it doesn't exist on disk
    transform=ToTensor(), # images come as PIL format, we want to turn into Torch tensors
    target_transform=None # you can transform labels as well
)

# Setup testing data
test_data = datasets.FashionMNIST(
    root="",
    train=False, # get test data
    download=True,
    transform=ToTensor()
)
pass