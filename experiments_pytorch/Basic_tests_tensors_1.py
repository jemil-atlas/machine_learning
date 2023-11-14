#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is to test absolute basic torch functionality of tensors. The goal 
here is to create a custom loss function built up of tensors and then to
minimize it.
For this, do the following:
    1. Imports and definitions
    2. Set up loss function
    3. Optimize 
    4. Illustrate results
"""

"""
    1. Imports and definitions
"""


# i) Imports

import torch
import matplotlib.pyplot as plt



"""
    2. Set up loss function
"""


# i) Initial tensors

# initialize a variable x. Here, x is a tensor with requires_grad=True which indicates to Autograd 
# that it needs to compute gradients with respect to these tensors during the backward pass.
x = torch.tensor([1.0], requires_grad = True)



"""
    3. Optimize 
"""


# i) Define Adam optimizer

# Adam is a popular optimization algorithm used in deep learning models and it has several advantages 
# over the classical stochastic gradient descent.
optimizer = torch.optim.Adam([x], lr=0.01)
x_history = []


# ii) Optimize

for step in range(500):
    # set the gradients to zero before starting to do backpropragation because 
    # PyTorch accumulates the gradients on subsequent backward passes.
    optimizer.zero_grad()  
    
    # compute the loss function
    y = x**2
    loss = torch.abs(y)
    
    # compute the gradients
    loss.backward()
    # update the weights
    optimizer.step()
    x_history.append(x.item())
    
    # print the loss value and the value of x at specifix steps
    if step % 100 == 0:
        print(f"Step {step+1}, Loss {loss.item()}, x {x.item()}")



"""
    4. Illustrate results
"""

# i) Plot history of x adjustments

plt.figure(1, dpi = 300)
plt.plot(x_history)
plt.title('Sequence of x values')

