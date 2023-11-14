#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides functionality for showcasing code and functionality of cvxpy,
a classic python library for optimization. This is done by fitting some model
coefficients subject to linear equalities and inequalities. 
For this, do the following:
    1. Imports and definitions
    2. Generate some data
    3. Create the ANN
    4. Train and assemble the solution
    5. plots and illustrations        
    
Author: Dr. Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.ch.
Copyright: 2021-2022, Atlas optimization GmbH, Zurich, Switzerland. All rights reserved.
"""

"""
    1. Imports and definitions
"""


# i) Imports

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader

# ii) Definitions

n_t = 100
time = np.linspace(0,1,n_t)
np.random.seed(1)



"""
    2. Generate some data
"""

# i) True model

true_coeff = np.array([[0.0],[0.1],[0.3]])
A = np.vstack((np.ones(n_t), time, np.sin(2*np.pi*time))).T

noise = np.reshape(np.random.normal(0,0.2,n_t),[n_t,1])
data = (A@true_coeff + noise).flatten()

full_data = torch.tensor(np.vstack((time, data)))

# ii) Create data loaders

batch_size = 64
train_dataloader = DataLoader(full_data, batch_size=batch_size)





"""
    3. Create the ANN
"""


# i) NN class

device = 'cpu'
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        fun_val = self.linear_relu_stack(x)
        return fun_val

model = NeuralNetwork().to(device)
print(model)



"""
    4. Train and assemble solution
"""


# i) Define loss

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)


# ii) Define training function

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device).reshape([n_t,1]).float(), y.to(device).reshape([n_t,1]).float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
 
            
# iii) Train

epochs = 5000
for t in range(epochs):
    # print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
print("Done!")


# iv) Build predictions

data_hat = model(torch.tensor(time).reshape([n_t,1]).float())



"""
    5. plots and illustrations  
"""

# i) Data

plt.figure(1, dpi=300)
sns.scatterplot(x=time,y=data.flatten()).set(title = 'data and model fit', xlabel = 'time', ylabel = 'fun value')
sns.lineplot(x=time, y=(data_hat.detach().numpy()).flatten(), label = 'ann')
plt.legend()



































