#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to illustrate training an ml model to learn some 
correction functions for tls data. These functions should allow converting
intensities associated to range r_1 and angle of incidence (aoi) phi_1 to a 
hypothetical intensity observed at (r_2, phi_2).
For this, do the following:
    1. Definitions and imports
    2. Simulate some data
    3. Build model
    4. Inference with torch
    5. Plots and illustrations

We use simulated data that comes from a simple model, in which intensity decreases
monotonically with range and aoi. There is only one material class and we employ
a simple pytorch model that converts a constant material property A to observed
intensity values via the separable equation I = f(r) g(phi) * A. In a later iteration,
we employ pyro to jointly estimate these conversion functions and material classes.

Please note the loss function: \| I_k * (f(r_l)g(phi_l)/f(r_k)g(phi_k)) - I_l\|
reflects the convertability of intensity measurements associated to different
locations. The measurement setup looks like this (. = point, X = station):

    (-1,1) ............................................... (1,1)
    
    
            
              (-0.5,0) X                 X (0.5,0)

The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""


"""
    1. Definitions and imports
"""


# i) Imports

import torch
import matplotlib.pyplot as plt


# ii) Definitions

n_points = 100
n_stations = 2


"""
    2. Simulate some data
"""

# i) Set up true conversion function

alpha_true = 1
beta_true = 1

def f_true(r):
    return torch.exp(- alpha_true *r)

def g_true(phi):
    return torch.cos(beta_true * phi)

def cartesian_to_polar(x,y):
    r = torch.sqrt(x**2 + y**2)
    phi = torch.atan2(x,y)
    return r, phi
    

# ii) Space data points along line

A_material = 1
x_points = torch.linspace(-1, 1, n_points)
y_points = torch.linspace(1, 1, n_points)

r_points = torch.zeros([n_stations,n_points])
phi_points = torch.zeros([n_stations,n_points])

f_vals = torch.zeros([n_stations,n_points])
g_vals = torch.zeros([n_stations,n_points])


# iii) Simulate data for stations

x_stations = torch.linspace(-0.5, 0.5, n_stations)
y_stations = torch.linspace(-0.0, 0.0, n_stations)

intensity_data = torch.zeros([n_stations, n_points])

for k in range(n_stations):
    for l in range(n_points):
        # Compute data entries
        r_temp, phi_temp = cartesian_to_polar(x_points[l] - x_stations[k], 
                                              y_points[l] - y_stations[k])        
        intensity_temp = A_material*f_true(r_temp)*g_true(phi_temp)
        intensity_data[k,l] = intensity_temp
        
        # Quantities for illustration
        f_temp, g_temp = f_true(r_temp), g_true(phi_temp)
        r_points[k,l] = r_temp
        phi_points[k,l] = phi_temp
        f_vals[k,l] = f_temp
        g_vals[k,l] = g_temp
        
noise = torch.distributions.Normal(0,0.01).sample([n_stations, n_points])
intensity_data_noisy = intensity_data + noise



"""
    3. Build model
"""


# i) Neural network class

class ANN(torch.nn.Module):
    def __init__(self, dim_hidden):
        # Initialize ANN class by initializing the superclass
        super().__init__()
        self.dim_input = 1
        self.dim_output = 1
        self.dim_hidden = dim_hidden
        
        # linear transforms
        self.fc_1 = torch.nn.Linear(self.dim_input, dim_hidden)
        self.fc_2 = torch.nn.Linear(dim_hidden, dim_hidden)
        self.fc_3 = torch.nn.Linear(dim_hidden, self.dim_output)
        # nonlinear transforms - produces positive numbers
        self.nonlinear = torch.nn.Sigmoid()
        
    def forward(self, input_data):
        # Define forward computation on the input input_data
        # Shape the minibatch so that batch_dims are on left
        input_data = input_data.reshape([-1, 1])
        
        # Then compute hidden units and output of nonlinear pass
        hidden_units_1 = self.nonlinear(self.fc_1(input_data))
        hidden_units_2 = self.nonlinear(self.fc_2(hidden_units_1))
        output = self.nonlinear(self.fc_3(hidden_units_2))
        # The output is a scaling coefficient for each input
        return output


# ii) Initialize f and g conversion functions

# f and g are both ANN's with n_hidden hidden dims
n_hidden = 10
f_model = ANN(n_hidden)
g_model = ANN(n_hidden)



"""
    4. Inference with torch
"""


 # i) Set up optimizer and loss

params = list(f_model.parameters()) + list(g_model.parameters())
optimizer = torch.optim.Adam(params, lr=0.01)           
loss_history = []

def loss_fn(data, f_model, g_model):
    
    eps = 1e-3
    
    I_obs = data.reshape([-1,1])
    conv_coeffs = f_model(r_points.reshape([-1,1])) * g_model(phi_points.reshape([-1,1]))
    # conversion_ratio = conv_coeffs / (conv_coeffs.T + eps)
    conversion_ratio = 1/(conv_coeffs + eps) * (conv_coeffs.T)
    
    I_pred = I_obs * conversion_ratio
    
    diff_mat = I_obs - I_pred.T 
    loss_val = torch.norm(diff_mat, p = 'fro')
    return loss_val


# ii) Optimize

for epoch in range(1000):
    # Set the gradients to zero 
    optimizer.zero_grad()  
    
    # compute the loss function
    loss = loss_fn(intensity_data_noisy, f_model, g_model) 
    
    # compute the gradients
    loss.backward()
    # update the weights, record new parameters and loss
    optimizer.step()
    loss_history.append(loss.item())
        
    # print the loss value at specifix steps
    if epoch % 100 == 0:
        print("Epoch {}, Loss = {}".format(epoch+1, loss.item()))



"""
    5. Plots and illustrations
"""


# i) Illustration of data and ground truth

fig, ax = plt.subplots(4,1, figsize = (5,15), dpi = 300)

# geometric configuration
ax[0].scatter(x_points, y_points, label = 'points')
ax[0].scatter(x_stations, y_stations, label = 'stations')
ax[0].legend()
ax[0].set_title('Measurement setup')

# data
ax[1].plot(intensity_data_noisy.T, label = 'intensity_data')
ax[1].set_title('Observed intensity data')

# range and aoi
ax[2].plot(r_points.T, label = 'range')
ax[2].plot(phi_points.T, label = 'aoi')
ax[2].legend()
ax[2].set_title('Range and AOI of points')

# conversion functions f,g
ax[3].plot(f_vals.T, label = 'distance effect f')
ax[3].plot(g_vals.T, label = 'aoi effect g')
ax[3].legend()
ax[3].set_title('Conversion functions f,g')


# ii) Illustration of training progress

plt.figure(num =2, figsize = (10,5), dpi = 300)
plt.plot(loss_history)
plt.title('Training loss')
plt.show()


# iii) Illustration of learned functions

f_vals_model = f_model(r_points).reshape([n_stations, n_points])
g_vals_model = g_model(phi_points).reshape([n_stations, n_points])
predicted_data = A_material*f_vals_model * g_vals_model


fig, ax = plt.subplots(4,1, figsize = (5,15), dpi = 300)

# true conversion functions
ax[0].plot(f_vals.T, label = 'distance effect f')
ax[0].plot(g_vals.T, label = 'aoi effect g')
ax[0].legend()
ax[0].set_title('True functions f,g')

# learned functions
ax[1].plot(f_vals_model.detach().T, label = 'distance effect f')
ax[1].plot(g_vals_model.detach().T, label = 'aoi effect g')
ax[1].legend()
ax[1].set_title('Learned functions f,g')

# true data
ax[2].plot(intensity_data_noisy.T, label = 'intensity_data')
ax[2].set_title('Observed intensity data')

# predicted data using learned functions
ax[3].plot(predicted_data.detach().T, label = 'intensity_data')
ax[3].set_title('Predicted intensity data')









    
    