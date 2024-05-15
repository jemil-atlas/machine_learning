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
import copy
import matplotlib.pyplot as plt
from pyro.infer import config_enumerate


# ii) Definitions

n_points = 100
n_stations = 2
n_materials_true = 1
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

class EnumANN(torch.nn.Module):
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
        
    def forward(self, s, z):
        # Input args: 
        #   - s : Tensor of shape [n_obs, n_stations] represents either r or phi
        #   - z : Tensor of shape [n_obs, 1] or [n_enumerate, 1, 1]
        # Output has shape [1, n_obs, n_stations] or is batched by enumeration
        # to [n_enumerate, n_obs, n_stations]

        # Reshaping operations        
    
        # Convert z to unified [n_enumerate, n_obs, n_stations] where n_enumerate is 1 during simulation
        if z.dim() == 2:
            z = z.unsqueeze(0)  # [n_obs,1] -> [1, n_obs,1]    
        # Convert z to [n_enumerate, n_obs, n_stations]
        z_reshaped = z.expand(-1,s.size(-2),s.size(-1))
        # Convert s to [n_enumerate, n_obs, n_stations]
        s_reshaped = s.unsqueeze(0).expand(z_reshaped.size(0), -1,-1)
        # Combine to arg of shape [n_enumerate, n_obs, n_station, 2]
        arg = torch.stack((s_reshaped,z_reshaped), dim = -1)
        
        # Compute hidden units and output of nonlinear pass
        arg_reshaped = arg.view(-1,2)
        hidden_units_1 = self.nonlinear(self.fc_1(arg_reshaped))
        hidden_units_2 = self.nonlinear(self.fc_2(hidden_units_1))
        output = self.nonlinear(self.fc_3(hidden_units_2))
        output_reshaped = output.view(z_reshaped.size(0), s_reshaped.size(1), s_reshaped.size(2))
        # The output is a scaling coefficient for each input
        
        return output_reshaped


# ii) Initialize f and g conversion functions

# f and g are both ANN's with n_hidden hidden dims
n_hidden = 10
f_model_net = EnumANN(2,n_hidden,1)
g_model_net = EnumANN(2,n_hidden,1)

def f_model(r,z):
    # Input shape is    r = [ n_obs, n_stations, 1]
    #                   z = [n_enumerate, n_obs, n_stations, 1]
    # Output shape is   result = [n_enumerate, n_obs, n_stations, 1]
    result = f_model_net(r, z)
    return result

def g_model(phi, z):
    # Input shape is    phi = [n_obs, n_stations, 1]
    #                   z = [n_enumerate, n_obs, n_stations, 1]
    # Output shape is   result = [n_enumerate, n_obs, n_stations, 1]
    result = f_model_net(phi, z)
    return result


# iii) Construct the model

@config_enumerate
def model(geometry, observations = None):
    # Initializations, shapes and parameters
    pyro.module('f_ann', f_model_net)
    pyro.module('g_ann', g_model_net)
    
    n_obs = geometry.shape[0]
    
    # parameter setup    
    rel_probs = pyro.param('rel_probs', (1/n_materials)*torch.ones(n_materials),
                           pyro.distributions.constraints.simplex)
    sigma = 0.01 * torch.eye(1)
    # sigma = pyro.param('sigma', 0.01*torch.eye(1), constraint = pyro.distributions.constraints.positive)

    # Local variables
    with pyro.plate('batch_plate_obs', size = n_obs, dim = -2) as ind_o:
        latent_z_dist = pyro.distributions.Categorical(probs = rel_probs).expand([n_obs,1])
        latent_z = pyro.sample('latent_z', latent_z_dist)
        
        with pyro.plate('batch_plate_stations', size = n_stations, dim = -1) as ind_s:            
            r = geometry[ind_o.unsqueeze(-1),ind_s.unsqueeze(-2),0]
            phi = geometry[ind_o.unsqueeze(-1),ind_s.unsqueeze(-2),1]
            
            f_vals = f_model(r, latent_z)
            g_vals = g_model(phi, latent_z)
            intensity = A_material * f_vals*g_vals
            
            obs_dist = pyro.distributions.Normal(loc = intensity, scale = sigma)
            obs = pyro.sample('obs', obs_dist, obs = observations)
            
            # # Diagnosis
            # print("latent_z.shape = {}".format(latent_z.shape))
            # print("latent_z_dist.batch_shape = {}".format(latent_z_dist.batch_shape))
            # print("obs.shape = {}".format(obs.shape))
            # print("obs_dist.batch_shape = {}".format(obs_dist.batch_shape))
            # print("obs_dist.shape = {}".format(obs_dist.shape()))
        
        return obs, latent_z
    

# iv) Construct the guide

def guide(geometry, observations = None):
    pass



"""
    4. Inference with pyro
"""

# i) Pyro inference

adam = pyro.optim.NAdam({"lr": 0.01})
elbo = pyro.infer.TraceEnum_ELBO(max_plate_nesting = 2)
svi = pyro.infer.SVI(model, guide, adam, elbo)

# Diagnosis
print("Sampling:")
_ = model(geometry, intensity_data_noisy)
print("Enumerated Inference:")
_ = elbo.loss(model, guide, *(geometry, intensity_data_noisy))

loss_sequence = []
for step in range(2000):
    loss = svi.step(geometry, intensity_data_noisy)
    if step % 100 == 0:
        print('epoch: {} ; loss : {}'.format(step, loss))
    else:
        pass
    loss_sequence.append(loss)


# ii) Simulate from trained model
simulation_trained, latent_z_trained = copy.copy(model(geometry))
simulation_trained = simulation_trained.detach()
latent_z_trained = latent_z_trained.detach()



"""
    5. Plots and illustrations
"""


# i) Illustration of data and ground truth

fig, ax = plt.subplots(4,1, figsize = (5,15), dpi = 300)

# geometric configuration
for k in range(n_materials_true):    
    ax[0].scatter(x_points[class_indices[k]], y_points[class_indices[k]], label = 'points mclass {}'.format(k))
ax[0].scatter(x_stations, y_stations, label = 'stations')
ax[0].legend()
ax[0].set_title('Measurement setup')

# data
ax[1].plot(intensity_data_noisy, label = 'intensity_data')
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
plt.plot(loss_sequence)
plt.title('Training loss')
plt.show()


# iii) Illustration of learned functions

f_vals_model = f_model(r_points.T, latent_z_trained).reshape([n_points, n_stations])
g_vals_model = g_model(phi_points.T, latent_z_trained).reshape([n_points, n_stations])
predicted_data = A_material*f_vals_model * g_vals_model


fig, ax = plt.subplots(4,1, figsize = (5,15), dpi = 300)

# true conversion functions
ax[0].plot(f_vals.T, label = 'distance effect f')
ax[0].plot(g_vals.T, label = 'aoi effect g')
ax[0].legend()
ax[0].set_title('True functions f,g')

# learned functions
ax[1].plot(f_vals_model.detach(), label = 'distance effect f')
ax[1].plot(g_vals_model.detach(), label = 'aoi effect g')
ax[1].legend()
ax[1].set_title('Learned functions f,g')

# true data
ax[2].plot(intensity_data_noisy, label = 'intensity_data')
ax[2].set_title('Observed intensity data')

# predicted data using learned functions
ax[3].plot(predicted_data.detach(), label = 'intensity_data')
ax[3].set_title('Predicted intensity data')





