# PyTorch Weather Classification

## Objective
Developing a Convolutional Neural Network (CNN) capable of matching (to start) ~85% accuracy with a select group of weather phenomena. This will be converted into an application and applied both on my personal website and for later use elsewhere. Some of the code will be partially from PyTorch's tutorials, such as "https://pytorch.org/tutorials/beginner/data_loading_tutorial.html" and "https://pytorch.org/tutorials/beginner/basics/data_tutorial.html".

## Statement
This original project was conducted using TensorFlow and is available on my GitHub under 'CNN-Django-Application', which does a similar job as this, but is less configurable.

## Datasets
The dataset is a collection of several Kaggle datasets and personal images taken over time. the reasoning behind combining datasets is to provide the model with several different images. Several different sizes of image also exist, giving my an opportunity to dive into the deep end of PyTorch.

## Current work
This will be paired with Object Detection eventually, combining the CNN with a method for identifying objects in images (as that is very common in these photos). This is a continuous project that will be on going, alongside my NLP work in another repo.

## Goals
This is going to be a model that is 'portable', meaning it's available for launching on applications using Django or any other application-based libraries.

## Methods and Libraries
The purpose of this project is to apply PyTorch fundamentals along side CNN capabilities to classify the environment around us. Several libraries are used, but the main core are listed below:
* PyTorch
* Pandas
* Matplotlib
* Seaborn
* NumPy
* PIL
* os

## Future work
I plan on getting this spun up in a Docker container, so an expected dockerfile is to be attached with this in the near future.

