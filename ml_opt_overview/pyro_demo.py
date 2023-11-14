#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides functionality for showcasing code and functionality of pyro,
a python library for probabilistic programming. This is done by performing mcmc
based inference on a task dealing with measurement uncertainty. The exact task
consists in a construction elements length being measured; there is a prior on
the elements length (based on the production process) and a distribution based
on measurement uncertainty. 
For this, do the following:
    1. Imports and definitions
    2. Generate some data
    3. Formulate the stochastic model
    4. Solve the model with mcmc
    5. Plots and illustrations        
    
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
import pyro
import pyro.distributions as dist
from pyro.infer import NUTS, MCMC


# ii) Definitions

n_sample = 100
np.random.seed(1)

n_d = 100



"""
    2. Generate some data
"""


# i) True length

mean_length = 1
var_length = 0.1
length = torch.tensor(np.random.normal(mean_length, var_length))


# ii) Observations

mean_obs = mean_length
var_obs =0.01

observations = torch.tensor(np.random.normal(length,var_obs,[n_sample]))



"""
    3. Formulate the stochastic model
"""


# i) Invoke model

def stoch_model(data=None):
    theta = pyro.sample("theta", dist.Normal(mean_length, var_length))
    
    with pyro.plate("data",n_sample):
        return pyro.sample("obs",dist.Normal(theta,var_obs),obs=data)


# ii) Set up mcmc

n_simu_hmc=800
nuts_kernel=NUTS(model = stoch_model)         # Build hmc kernel
mcmc_results=MCMC(nuts_kernel,num_samples=n_simu_hmc,warmup_steps=200) # Build hmc setup



"""
    4. Solve the model with mcmc
"""


# Run mcmc

mcmc_results.run(observations)         # Run hmc and populate samples

samples = mcmc_results.get_samples()
point_estimate=torch.mean(samples['theta'])



"""
    5. Plots and illustrations  
"""


# i) Data

plt.figure(1, dpi=300)
sns.histplot(data = samples, stat = 'density', kde = True, label = 'posterior density').set(title = 'posterior density', xlabel = 'length element', ylabel = 'rel frequency')
sns. scatterplot(x= [length], y = [500], color = 'r', label = 'true value')
plt.legend()

































