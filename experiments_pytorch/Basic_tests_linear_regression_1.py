#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script provides a simple example of linear regression being trained using
pytorch. For this, do the following:
    1. Imports and definitions
    2. Generate data
    3. Build model and train
    4. Plots and illustrations
"""

"""
    1. Imports and definitions
"""


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt



"""
    2. Generate data
"""


# Assuming we have input features of dimension 5
input_dim = 1

# Assume we have some data in the following variables
# X is our features, of shape (num_samples, input_dim)
# y is our target variable, of shape (num_samples, 1)
X = torch.randn(100, input_dim)
y = torch.randn(100, 1)


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
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# A simple training loop
for epoch in range(100):
    model.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
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
plt.scatter(X[:,0],y)
plt.plot(x_test.detach(), y_test.detach())




