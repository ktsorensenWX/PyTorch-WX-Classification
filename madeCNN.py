####################################################################
# Packages
import torch
import torch.nn as nn
import torch.nn.functional as F

####################################################################
# Model

"""
Structure: --> my structure to build here
    1) Conv 1 - Valid padding; kernel (5x5)
    2) Pool (Max) 1 
    3) Conv 2 - Valid padding; kernel (5x5)
    4) Pool (Max) 2
    5) Conv 3 - Valid padding; kernel (5x5)
    6) Pool (Max) 3
    7) Dense (FCN) w/ ReLU
    8) Output w/ Softmax
""";

# Define the net - Import this into the jupyter notebook - dummy/test net
class Net(nn.Module):
    # Define the settings of each layer
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(44944, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 6)

    # Set the network in motion
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # Applying ReLU activation, then pooling the conv layer (can also break this out
        x = self.pool(F.relu(self.conv2(x))) # into individual layers)
        x = torch.flatten(x, 1)              # flatten all dimensions except batch
        x = F.relu(self.fc1(x))              # ReLU on the fcn
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))           # Applying softmax on the output                      
        return x

class KtsNet(nn.Module):
    # Define the settings of each layer according to the structure
    def __init__(self):
        super().__init__()
        self.conv1 = ...
        self.pool1 = ...
        self.conv2 = ...
        self.pool2 = ...
        self.conv3 = ...
        self.pool3 = ...
        self.fc1 = ...
        self.fc2 = ...
        self.fc3 = ...

    # Set the network in motion
    def forward(self, x):                   
        return x

# Testing linkage
def tester(x):
    print(x)