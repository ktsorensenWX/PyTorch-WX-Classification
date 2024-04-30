####################################################################
# Packages
import torch
import torch.nn as nn
import torch.nn.functional as F

####################################################################
# Model

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

# Testing linkage
def tester(x):
    print(x)