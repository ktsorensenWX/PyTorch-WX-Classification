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