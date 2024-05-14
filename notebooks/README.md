# PyTorch Weather Classification Notebooks

## Objective
These notebooks include the data collection, manipulation, exploration, and other methodologies. The original notebook "wx_PyTorch" contains models including ResNet34, GoogLeNet, and a basic CNN. The Dataloaders here however are "bottlenecked", meaning when ran, the training is very slow. The reason for this is the introduction of several other functions that alter the size and color of the images. By manually creating and using these functions, we run into an issue where the batches as they're loaded into the model, go through so many hoops that loading is just not fast, no matter what GPU you have. 

The other notebook, "wx_new" revamps that previous one, making use of PyTorch's built-in resizing and normalizing tools to help speed up the process. The training results prove to be much better (accuracy-wise) and quicker (15 epochs ~ 2 minutes versus the previous 40).

## Goals
I'll be calling the models from these notebooks. Each will use some different way of loading or storing the data (i.e. the Dataset/Dataloaders). Purpose of this is to get a better understanding of speed within PyTorch. Certain mechanics are much faster than others, however, you can lose some functionality & configurability with speed.

## Methods and Libraries
The purpose of this project is to apply PyTorch fundamentals along side CNN capabilities to classify the environment around us. Several libraries are used, but the main core are listed below:
* PyTorch
* Tensorboard
* Pandas
* Matplotlib
* NumPy
* PIL
* os

