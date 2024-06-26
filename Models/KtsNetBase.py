####################################################################
# Packages
import torch
import torch.nn as nn
import torch.nn.functional as F

####################################################################
# Model v1

"""
Structure: --> my structure to build here
    1) Conv 1 - Valid padding; kernel (5x5)
    2) Pool (Max) 1 
    3) Conv 2 - Valid padding; kernel (5x5)
    4) Pool (Max) 2
    5) Conv 3 - Valid padding; kernel (5x5)
    6) Pool (Max) 3
    7) Dense (FCN) w/ ReLU
    8) Dence (FCN) output

Model v1 will have a MAX of 3 sequential layers and 3 max pool layers
""";

class KtsNet_v1(nn.Module):
    # Define the settings of each layer according to the structure
    def __init__(self, in_channels, num_classes=0):
        super().__init__()

        # Conv1 will preform a convolution with ReLU activation & Batch Norm
        self.conv1 = nn.Sequential(
                            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                            nn.BatchNorm2d(32),
                            nn.ReLU(),
                        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv2 will preform a convolution with ReLU activation
        self.conv2 = nn.Sequential(
                            nn.Conv2d(32, 64, kernel_size=3, padding=1),
                            nn.ReLU(),
                        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv3 will preform a convolution with ReLU activation
        self.conv3 = nn.Sequential(
                            nn.Conv2d(64, 128, kernel_size=3, padding=1),
                            nn.ReLU(),
                        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Now the linear/Flattening Layers
        self.fc1 = nn.Sequential(
                            nn.Flatten(),
                            nn.Linear(64 * 28 * 28, 512),
                            nn.ReLU()
                        )
        self.fc2 = nn.Linear(512, num_classes)

    # Set the network in motion
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

####################################################################
# Model v2

"""
Structure: --> my structure to build here
    1) Conv 1 - Valid padding; kernel (5x5)
    2) Pool (Max) 1 
    3) Conv 2 - Valid padding; kernel (5x5)
    4) Pool (Max) 2
    5) Conv 3 - Valid padding; kernel (5x5)
    6) Pool (Max) 3
    7) Conv 4 - Valid padding; kernel (5x5)
    8) Pool (Max) 4
    9) Dense (FCN) w/ ReLU
    10) Dence (FCN) output
""";

# Model will incorporate deeper layers for finer feature extractions
class KtsNet_v2(nn.Module):
    # Define the settings of each layer according to the structure
    def __init__(self, in_channels, num_classes=0):
        super().__init__()

        # Conv1 will preform a convolution with ReLU activation & Batch Norm
        self.conv1 = nn.Sequential(
                            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                            nn.BatchNorm2d(32),
                            nn.ReLU(),
                        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv2 will preform a convolution with ReLU activation
        self.conv2 = nn.Sequential(
                            nn.Conv2d(32, 64, kernel_size=3, padding=1),
                            nn.ReLU(),
                        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv3 will preform a convolution with ReLU activation
        self.conv3 = nn.Sequential(
                            nn.Conv2d(64, 128, kernel_size=3, padding=1),
                            nn.ReLU(),
                        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv4 will preform a convolution with ReLU activation
        self.conv4 = nn.Sequential(
                            nn.Conv2d(128, 256, kernel_size=3, padding=1),
                            nn.ReLU(),
                        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Now the linear/Flattening Layers
        self.fc1 = nn.Sequential(
                            nn.Flatten(),
                            nn.Linear(64 * 28 * 28, 512),
                            nn.ReLU()
                        )
        self.fc2 = nn.Linear(512, num_classes)

    # Set the network in motion
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x
    
####################################################################
# Model v3

"""
Structure: --> my structure to build here
    1) Conv 1 - Valid padding; kernel (5x5)
    2) Pool (Max) 1 
    3) Conv 2 - Valid padding; kernel (5x5)
    4) Pool (Max) 2
    5) Conv 3 - Valid padding; kernel (5x5)
    6) Pool (Max) 3
    7) Conv 4 - Valid padding; kernel (5x5)
    8) Pool (Max) 4
    9) Dense (FCN) w/ ReLU
    10) Dence (FCN) output
""";

class KtsNet_v3(nn.Module):
    # Define the settings of each layer according to the structure
    def __init__(self, in_channels, num_classes=0):
        super().__init__()

        # Conv1 will preform a convolution with ReLU activation & Batch Norm
        self.conv1 = nn.Sequential(
                            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                            nn.BatchNorm2d(32),
                            nn.ReLU(),
                        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv2 will preform a convolution with ReLU activation
        self.conv2 = nn.Sequential(
                            nn.Conv2d(32, 64, kernel_size=3, padding=1),
                            nn.ReLU(),
                        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv3 will preform a convolution with ReLU activation
        self.conv3 = nn.Sequential(
                            nn.Conv2d(64, 128, kernel_size=3, padding=1),
                            nn.ReLU(),
                        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Now the linear/Flattening Layers
        self.fc1 = nn.Sequential(
                            nn.Flatten(),
                            nn.Linear(128 * 28 * 28, 512),
                            nn.ReLU()
                        )
        self.fc2 = nn.Linear(512, num_classes)

    # Set the network in motion
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x