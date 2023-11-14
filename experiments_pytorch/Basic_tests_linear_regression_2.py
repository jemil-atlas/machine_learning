#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script provides a simple example of linear regression being trained using
pytorch. In comparison to the more basic 'Basic_tests_linear_regression_1', we 
will define our own custom loss function and optimize that one.
For this, do the following:
    1. Imports and definitions
    2. Generate data
    3. Build model and train
    4. Plots and illustrations
"""

"""
    1. Imports and definitions
"""


# i) Imports

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# ii) Definitions

input_dim = 1
n_data = 100



"""
    2. Generate data
"""



# i) Generate new data
# x is our features, of shape (n_data, input_dim)
# y is our target variable, of shape (n_data, 1)
mu_fun = lambda x : x
sigma_fun = lambda x : 0.1

x_data = torch.linspace(-1,1, n_data).reshape([-1,1]).float()
y_true = mu_fun(x_data)
sigma_true = sigma_fun(x_data)
y_data = torch.tensor(np.random.normal(y_true, sigma_true)).float()


# ii) Fuse data together and create new dataset

xy_data = torch.hstack((x_data,y_data)).float()
xy_tensor_data = torch.utils.data.TensorDataset(x_data, y_data)


"""
    3. Build model and train
"""


# Define a simple linear regression model
class LinearRegression(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegression(input_dim)

# We'll use mean squared error as our loss function, and Adam as our optimizer
def loss_fun(x, y):
    loss = torch.norm(model(x)- y, p=2)
    return loss

optimizer = optim.Adam(model.parameters(), lr=0.1)

# A simple training loop
for epoch in range(100):
    model.zero_grad()
    loss = loss_fun(x_data,y_data)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 100, loss.item()))


"""
    4. Plots and illustrations
"""


# Generate regular grid of x, then look at predictions
n_simu = 100
x_test = torch.linspace(-2,2,n_simu).reshape([-1,1])
y_test = model(x_test)
plt.figure(1, dpi = 300)
plt.scatter(x_data[:,0],y_data)
plt.plot(x_test.detach(), y_test.detach())




