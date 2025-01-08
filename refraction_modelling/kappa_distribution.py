#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to model the impact on kappa of single wavelengths 
being replaced by a distribution of wavelengths. How exactly replacing lambda_1,
lambda_2 by p(lambda_1), p(lambda_2) affects k(lambda_1, lambda_2) is not easy
to establish because:
        1. Distribution of lambda -> Distribution of beta [how to convert to single number?]
        2. The basic beta/mu(lambda) = const approach does not hold anymore [Boeckem, p.27]
        3. What do the operations [addition multiplication, division etc] mean for distributions?
            Do they actually solve the equations like they are supposed to?
These questions are not conclusively answered here; instead we will make the 
assumption that lamba_1, lambda_2 are drawn from probability distributions, then
converted to the k factor which itself becomes a random variable. The distribution
of the kappa random variable showcases possible
This might not be realistic but is one approach to computing the spread and allows
us to sidestep specifying answers to questions 1. - 3.

To perform this modelling, do the following:
    1. Definitions and imports
    2. Declare distributions for lambda
    3. Sample from these distributions
    4. Process towards kappa
    5. Plots and illustrations
"""



"""
    1. Definitions and imports
"""


# i) Imports

import torch
import pyro
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)


# ii) Definitions

n_simu = 10000
n_disc = 1000

lambda_1_mean = 0.430   # in mu m
lambda_2_mean = 0.860   # in mu m

lambda_1_sigma = 0.050
lambda_2_sigma = 0.050

lambda_spread = 0.2



"""
    2. Declare distributions for lambda
"""


# Notice that here we declare l1_p_vec via Normal pdf but it could be anything

# i) Distribution p(lambda_1)

lambda_1_dist = pyro.distributions.Normal(lambda_1_mean, lambda_1_sigma)
l1_range = torch.linspace(lambda_1_mean - lambda_spread, lambda_1_mean + lambda_spread, n_disc)
l1_p_vec = torch.exp(lambda_1_dist.log_prob(l1_range))
l1_p_vec = l1_p_vec / torch.sum(l1_p_vec)


# ii) Distribution p(lambda_2)

lambda_2_dist = pyro.distributions.Normal(lambda_2_mean, lambda_2_sigma)
l2_range = torch.linspace(lambda_2_mean - lambda_spread, lambda_2_mean + lambda_spread, n_disc)
l2_p_vec = torch.exp(lambda_2_dist.log_prob(l2_range))
l2_p_vec = l2_p_vec / torch.sum(l2_p_vec)


"""
    3. Sample from these distributions
"""


# i) Sampling function

def sample_range(l_range, l_p_vec):
    dist = pyro.distributions.Categorical(probs = l_p_vec)
    index_samples = dist.sample([n_simu])
    samples = l_range[index_samples]
    return samples


# ii) Perform sampling

l1_sample = sample_range(l1_range, l1_p_vec)
l2_sample = sample_range(l2_range, l2_p_vec)



"""
    4. Process towards kappa
"""

# i) Implement Edlen equation

def refraction_edlen(lumbda):
    # wavelength in dimension of mu m
    refractive_index = 1 + (1e-8)*(8342.13 + 2406030*(1/(130 - (1/lumbda**2))) 
                                    + 15997*(1/(38.9 - (1/lumbda**2))))
    return refractive_index

n1_sample = refraction_edlen(l1_sample)
n2_sample = refraction_edlen(l2_sample)


# ii) Convert to kappa

def kappa(n1, n2):
    kappa = (n1 - 1)/(n1-n2)
    return kappa

k1_sample = kappa(n1_sample, n2_sample)
k2_sample = kappa(n2_sample, n1_sample)



"""
    5. Plots and illustrations
"""


# i) Plots of distributions and samples

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

axs[0, 0].plot(l1_range, l1_p_vec, color='red')
axs[0, 0].set_title('Prob dist for lambda 1')

axs[0, 1].plot(l2_range, l2_p_vec, color='blue')
axs[0, 1].set_title('Prob dist for lambda 2')

axs[1, 0].hist(l1_sample.detach().numpy(), bins=30, color='red', alpha=0.7)
axs[1, 0].set_title('Histogram samples lambda 1')

axs[1, 1].hist(l2_sample.detach().numpy(), bins=30, color='blue', alpha=0.7)
axs[1, 1].set_title('Histogram samples lambda 2')

plt.tight_layout()
plt.show()


# ii) Plots of indixes of refractivity and kappa values

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

axs[0, 0].hist(n1_sample.detach().numpy(), bins=30, color='red', alpha=0.7)
axs[0, 0].set_title('Histogram samples n 1')

axs[0, 1].hist(n2_sample.detach().numpy(), bins=30, color='blue', alpha=0.7)
axs[0, 1].set_title('Histogram samples n 2')

axs[1, 0].hist(k1_sample.detach().numpy(), bins=30, color='red', alpha=0.7)
axs[1, 0].set_title('Histogram samples kappa 1')

axs[1, 1].hist(k2_sample.detach().numpy(), bins=30, color='blue', alpha=0.7)
axs[1, 1].set_title('Histogram samples kappa 2')

plt.tight_layout()
plt.show()