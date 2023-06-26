# Kod oparty o ćwiczenie "7.2 Stepping through nn.MaxPool2d()" z ebook "Zero to Mastery Learn PyTorch for Deep Learning"

import torch
from torch import nn
import matplotlib as plt
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import timeit

torch.manual_seed ( 42 )

print ( f"### ### 7.2 Stepping through nn.MaxPool2d()" )
print ( f"### ### from Zero to Mastery Learn PyTorch for Deep Learning" )

### Pytorch section
print ( "\n### Pytorch section" )
print ( f"{torch.__version__ = }, {torchvision.__version__ = }" )

### Data section
print ( "\n### Data section" )
# Setup training data
image = torch.randn ( size = ( 1 , 3 , 64 , 64 ) )


# Check out what's inside the image
print ( f"{image = }" )
print ( f"{image.shape = }" )

### CNN section
conv_layer = nn.Conv2d (
            in_channels = 3 ,
            out_channels = 10 ,
            kernel_size = 5 ,
            stride = 2 ,
            padding = 0 )


# Create a sample nn.MaxPoo2d() layer
max_pool_layer = nn.MaxPool2d ( kernel_size = 2 )

# Pass data through just the conv_layer
test_image_through_conv = conv_layer ( image )
print ( f"{test_image_through_conv.shape = }" )

# Pass data through the max pool layer
test_image_through_conv_and_max_pool = max_pool_layer ( test_image_through_conv )
print ( f"{test_image_through_conv_and_max_pool.shape = }" )
