import torch
from torch import nn
import matplotlib as plt
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import timeit

torch.manual_seed ( 42 )

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
            out_channels = 6 ,
            kernel_size = 4 ,
            stride = 1 ,
            padding = 0 )

### Main section
output = conv_layer ( image )
#print ( output )
print ( output.shape )