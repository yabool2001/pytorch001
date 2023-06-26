# Kod oparty o ćwiczenie "7.3 Setup a loss function and optimizer for model_2" z ebook "Zero to Mastery Learn PyTorch for Deep Learning"

import torch
from torch import nn
import matplotlib as plt
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import timeit

print ( f"### ### 7.3 Setup a loss function and optimizer for model_2" )
print ( f"### ### from Zero to Mastery Learn PyTorch for Deep Learning" )

### Private functions
def train_step ( model : torch.nn.Module , data_loader : DataLoader , loss_fn : torch.nn.Module , optimizer : torch.optim.Optimizer , accuracy_fn ) :
	train_loss , train_acc = 0 , 0
	for batch , ( X , y ) in enumerate ( data_loader ) :
		# 1. Forward pass
		y_pred = model ( X )
		# 2. Calculate loss
		loss = loss_fn ( y_pred , y )
		train_loss += loss
		train_acc += accuracy_fn ( y_true = y , y_pred = y_pred.argmax ( dim = 1 ) ) # Go from logits -> pred labels
		# 3. Optimizer zero grad
		optimizer.zero_grad ()
		# 4. Loss backward
		loss.backward ()
        # 5. Optimizer step
        optimizer.step ()
    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss = len ( data_loader )
    train_acc = len ( data_loader )
    print ( f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f} %" )


### Pytorch section
print ( "\n### Pytorch section" )
print ( f"{torch.__version__ = }, {torchvision.__version__ = }" )

### Dataset section
print ( "\n### Datasets section" )
# Setup training data
train_data = datasets.FashionMNIST (
    root = "" , # where to download data to?
    train = True , # get training data
    download = True , # download data if it doesn't exist on disk
    transform = ToTensor() , # images come as PIL format, we want to turn into Torch tensors
    target_transform = None # you can transform labels as well
)

# Setup testing data
test_data = datasets.FashionMNIST ( root = "" , train = False , download = True , transform = ToTensor () )

# Look at first training sample
image , label = train_data[0]
print ( f"{train_data.data.shape = }, {train_data.targets.shape = }" )
print ( f"{train_data.classes = }" )
print ( f"{image.shape = }" )

### DataLoader section - turns a large Dataset into a Python iterable of smaller chunks.
# Setup the batch size hyperparameter
print ( "\n### DataLoader section" )
BATCH_SIZE = 32
train_dataloader = DataLoader (
    train_data , # dataset to turn into iterable
    batch_size = BATCH_SIZE , # how many samples per batch?
    shuffle = True # shuffle data every epoch?
)
test_dataloader = DataLoader ( test_data , batch_size = BATCH_SIZE , shuffle = False # don't necessarily have to shuffle the testing data
)
# Check out what's inside the training dataloader
train_features_batch , train_labels_batch = next ( iter ( train_dataloader ) )
print ( f"{train_features_batch.shape = } , {train_labels_batch.shape = }" )

### CNN section
class FashionMNISTModelV2 ( nn.Module ) :
    """
    Model architecture copying TinyVGG from:
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__ ( self , input_shape: int , hidden_units: int , output_shape: int ) :
        super ().__init__ ()
        self.block_1 = nn.Sequential (
                        nn.Conv2d (
                            in_channels = input_shape ,
                            out_channels = hidden_units ,
                            kernel_size = 3 , # how big is the square that's going over the image?
                            stride = 1 , # default
                            padding = 1 ) , # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number
                        nn.ReLU () ,
                        nn.Conv2d (
                            in_channels = hidden_units ,
                            out_channels = hidden_units ,
                            kernel_size = 3 ,
                            stride = 1 ,
                            padding = 1 ) ,
                        nn.ReLU () ,
                        nn.MaxPool2d (
                            kernel_size = 2 ,
                            stride = 2 ) # default stride value is same as kernel_size
                        )
        self.block_2 = nn.Sequential (
                        nn.Conv2d (
                            hidden_units ,
                            hidden_units ,
                            3 ,
                            padding = 1 ) ,
                        nn.ReLU () ,
                        nn.Conv2d (
                            hidden_units ,
                            hidden_units ,
                            3 ,
                            padding = 1 ) ,
                        nn.ReLU () ,
                        nn.MaxPool2d ( 2 ) )
        self.classifier = nn.Sequential (
                        nn.Flatten () ,
                        # Where did this in_features shape come from?
                        # It's because each layer of our network compresses andchanges the shape of our inputs data.
                        nn.Linear (
                            in_features = hidden_units * 7 * 7 ,
                            out_features = output_shape ) )
    def forward ( self , x : torch.Tensor ) :
        x = self.block_1 ( x )  
        # print(x.shape)
        x = self.block_2 ( x )
        # print(x.shape)
        x = self.classifier ( x )
        # print(x.shape)
        return x
torch.manual_seed ( 42 )
model_2 = FashionMNISTModelV2 (
            input_shape = 1 ,
            hidden_units = 10 ,
            output_shape = len ( train_data.classes ) )
print ( f"{model_2 = }" )

# Setup loss and optimizer
loss_fn = nn.CrossEntropyLoss ()
optimizer = torch.optim.SGD ( params = model_2.parameters () , lr = 0.1 )