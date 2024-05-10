#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to illustrate training an ml model to learn some 
correction functions and classify material based on tls data. These functions 
should allow converting intensities associated to range r_1, angle of incidence
(aoi) phi_1 and material z_1 to a hypothetical intensity observed at (r_2,phi_2,z_2).
For this, do the following:
    1. Definitions and imports
    2. Simulate some data
    3. Build model and guide
    4. Inference with pyro
    5. Plots and illustrations

We use simulated data that comes from a simple model, in which intensity decreases
monotonically with range and aoi. There are n_materials material classes and we 
employ a simple pyro model that converts a constant material property A to observed
intensity values via the separable equation I = f(r,z) g(phi,z) * A.

Please note the likelihood function is chosen to encode the probability of jointly 
observing I_k and I_l and therefore dependent on the term:
    \| I_k * (f(r_l,z_l)g(phi_l,z_l)/f(r_k,z_k)g(phi_k,z_k)) - I_l\|
which reflects the convertability of intensity measurements associated to different
locations. The measurement setup looks like this (., : = point, X = station):

    (-1,1) ....................:::::::::::::::::::::::::::: (1,1)
    
    
            
              (-0.5,0) X                 X (0.5,0)

The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""



"""
    1. Definitions and imports
"""


# i) Imports

import torch
import pyro
import matplotlib.pyplot as plt
from pyro.infer import config_enumerate


# ii) Definitions

n_points = 100
n_stations = 2
n_materials_true = 2
n_materials = 2


"""
    2. Simulate some data
"""

# i) Set up true conversion function

# different params for different materials
alpha_true = torch.linspace(0.1,1,n_materials_true)
beta_true = torch.linspace(0.1,1,n_materials_true)

def f_true(r, mclass):
    return torch.exp(- alpha_true[mclass] *r)

def g_true(phi, mclass):
    return torch.cos(beta_true[mclass] * phi)

def cartesian_to_polar(x,y):
    r = torch.sqrt(x**2 + y**2)
    phi = torch.atan2(x,y)
    return r, phi
    

# ii) Space data points along line

A_material = 1
x_points = torch.linspace(-1, 1, n_points)
y_points = torch.linspace(1, 1, n_points)

mclass_points = torch.round(torch.linspace(0,n_materials_true-1, n_points)).long()
class_indices = []
for k in range(n_materials_true):
    class_indices.append(torch.where(mclass_points == k)[0])

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
        mclass_temp = mclass_points[l]
        intensity_temp = A_material*f_true(r_temp, mclass_temp)*g_true(phi_temp, mclass_temp)
        intensity_data[k,l] = intensity_temp
        
        # Quantities for illustration
        f_temp, g_temp = f_true(r_temp, mclass_temp), g_true(phi_temp, mclass_temp)
        r_points[k,l] = r_temp
        phi_points[k,l] = phi_temp
        f_vals[k,l] = f_temp
        g_vals[k,l] = g_temp
        
noise = torch.distributions.Normal(0,0.01).sample([n_stations, n_points])

# shape: intensity_data = [100,2] = [batch, n_stations]
#        geometry = [100,2,2] = [batch, n_stations, rphi]
intensity_data_noisy = (intensity_data + noise).T
geometry = torch.cat((r_points.T.unsqueeze(2),phi_points.T.unsqueeze(2)), dim = 2)



"""
    3. Build model and guide
"""


# i) Neural network class

class ANN(torch.nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output):
        # Initialize ANN class by initializing the superclass
        super().__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
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
        input_data = input_data.reshape([-1, self.dim_input])
        
        # Then compute hidden units and output of nonlinear pass
        hidden_units_1 = self.nonlinear(self.fc_1(input_data))
        hidden_units_2 = self.nonlinear(self.fc_2(hidden_units_1))
        output = self.nonlinear(self.fc_3(hidden_units_2))
        # The output is a scaling coefficient for each input
        return output


# ii) Initialize f and g conversion functions

# f and g are both ANN's with n_hidden hidden dims
n_hidden = 10
f_model_net = ANN(2,n_hidden,1)
g_model_net = ANN(2,n_hidden,1)

def f_model(f_input_rz):
    # input shape is [batch_shape, n_stations + 1]
    z_repeated = f_input_rz[:,-1].unsqueeze(1).repeat([1,n_stations])
    z_reshaped = z_repeated.reshape([-1,1])
    geometry_reshaped = f_input_rz[:,0:n_stations].reshape([-1,1])
    f_input_reshaped = torch.hstack((geometry_reshaped, z_reshaped))
    
    # z_repeated = f_input_rz[:,-1].repeat([1,n_stations]).T
    # geometry_reshaped = f_input_rz[:,0:n_stations].T.reshape([-1,1])
    # f_input_reshaped = torch.hstack((geometry_reshaped, z_repeated)) 
    
    result = f_model_net(f_input_reshaped).reshape([-1,n_stations])
    return result

def g_model(g_input_phiz):
    # input shape is [batch_shape, n_stations + 1]
    z_repeated = g_input_phiz[:,-1].unsqueeze(1).repeat([1,n_stations])
    z_reshaped = z_repeated.reshape([-1,1])
    geometry_reshaped = g_input_phiz[:,0:n_stations].reshape([-1,1])
    g_input_reshaped = torch.hstack((geometry_reshaped, z_reshaped))
    
    # z_repeated = g_input_phiz[:,-1].repeat([1,n_stations]).T
    # geometry_reshaped = g_input_phiz[:,0:n_stations].T.reshape([-1,1])
    # g_input_reshaped = torch.hstack((geometry_reshaped, z_repeated)) 
    
    result = g_model_net(g_input_reshaped).reshape([-1,n_stations])
    return result


# iii) Construct the model

@config_enumerate
def model(geometry, observations = None):
    # shapes
    n_obs = geometry.shape[0]
    
    # parameter setup    
    rel_probs = pyro.param('rel_probs', (1/n_materials)*torch.ones(n_materials),
                           pyro.distributions.constraints.simplex)
    sigma = pyro.param('sigma', torch.eye(1), constraint = pyro.distributions.constraints.positive)

    # Local variables
    with pyro.plate('batch_plate', size = n_obs, dim = -1):
        latent_z_dist = pyro.distributions.Categorical(probs = rel_probs).expand([n_obs])
        latent_z = pyro.sample('latent_z', latent_z_dist)
        
        f_input_rz = torch.hstack((geometry[:,:,0], latent_z.reshape([-1,1])))
        g_input_phiz = torch.hstack((geometry[:,:,1], latent_z.reshape([-1,1])))
        
        f_vals = f_model(f_input_rz)
        g_vals = g_model(g_input_phiz)
        intensity = A_material * f_vals*g_vals
        
        obs_dist = pyro.distributions.Normal(loc = intensity, scale = sigma).to_event(1)
        obs = pyro.sample('obs', obs_dist, obs = observations)
        
        return obs
    

# @config_enumerate
# def model(geometry, observations = None):
#     # shapes
#     n_obs = geometry.shape[0]
    
#     # parameter setup    
#     rel_probs = pyro.param('rel_probs', (1/n_materials)*torch.ones(n_materials),
#                            pyro.distributions.constraints.simplex)
#     sigma = pyro.param('sigma', torch.eye(1), constraint = pyro.distributions.constraints.positive)

#     # Local variables
#     with pyro.plate('batch_plate', size = n_obs, dim = -1):
#         latent_z_dist = pyro.distributions.Categorical(probs = rel_probs).expand([n_obs])
#         latent_z = pyro.sample('latent_z', latent_z_dist)
        
#         f_input_rz = torch.hstack((geometry[:,:,0], latent_z.reshape([-1,1])))
#         g_input_phiz = torch.hstack((geometry[:,:,1], latent_z.reshape([-1,1])))
        
#         f_vals = f_model(f_input_rz)
#         g_vals = g_model(g_input_phiz)
#         intensity = A_material * f_vals*g_vals
        
#         obs_dist = pyro.distributions.Normal(loc = intensity, scale = sigma).to_event(1)
#         obs = pyro.sample('obs', obs_dist, obs = observations)
        
#         return obs


# iv) Construct the guide

def guide(geometry, observations = None):
    pass



"""
    4. Inference with pyro
"""

# i) Pyro inference

adam = pyro.optim.NAdam({"lr": 0.01})
elbo = pyro.infer.TraceEnum_ELBO(max_plate_nesting = 1)
svi = pyro.infer.SVI(model, guide, adam, elbo)

loss_sequence = []
for step in range(1000):
    loss = svi.step(geometry, intensity_data_noisy)
    if step % 100 == 0:
        print('epoch: {} ; loss : {}'.format(step, loss))
    else:
        pass
    loss_sequence.append(loss)




"""
    5. Plots and illustrations
"""


# i) Illustration of data and ground truth

fig, ax = plt.subplots(4,1, figsize = (5,15), dpi = 300)

# geometric configuration
for k in range(n_materials):    
    ax[0].scatter(x_points[class_indices[k]], y_points[class_indices[k]], label = 'points mclass {}'.format(k))
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





