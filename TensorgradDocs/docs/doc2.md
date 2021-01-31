---
id: doc2
title: Layers
sidebar_label: Layers
slug: /layers
---

# Layers

## Activations

These layers are usually placed after a Weighted Layer, and add a nonlinearity to the function.

### Relu

This is the Relu activation. This equasion makes all negative values 0.

The equasion for this function is Relu(x) = max(0,x)


### Sigmoid

This is the Sigmoid activation. This function limits the input to in between 0 and 1.

The equasion for this function is Sigmoid(x) = e^x/e^x + 1

### Softmax

This is the Softmax activation. This function limits the input to a probability distribution.

The equasion for this is Softmax(x) =  e^x / sum(e^x,axis=-1)


## Weighted Layers


### Conv2d

This is the convolutional layer.

Args:

    inputs (int): The number of inputs to the convolution.
    outputs (int): The number of outputs to the convolution
    kernel_dim (int): The width and height of the kernel that will be applied to the image.
    stride (tuple, optional): The stride of which the kernel will be applied. Defaults to (1,1).
    use_bias (bool, optional): If true, adds a bias. Defaults to True.
    padding (string, optional): The padding around the image that will be convolved. Defaults to 'valid'.

Returns:


    Tensor: This is the output of the convolutional layer. These are the input images, with the kernel applied.



### Linear

This is the basic Neural Network layer.

### BatchNorm

## Non Weighted / Reshape Layers

### Reshape

### Flatten

### MaxPool2d