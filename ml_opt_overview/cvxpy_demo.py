#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides functionality for showcasing code and functionality of cvxpy,
a classic python library for optimization. This is done by fitting some model
coefficients subject to linear equalities and inequalities. 
For this, do the following:
    1. Imports and definitions
    2. Generate some data
    3. Create the CVXPY problem
    4. Assemble the solution
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
import cvxpy as cp


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


# ii) Standard LS

x_hat_ls = np.linalg.pinv(A)@data
data_hat_ls = A@x_hat_ls



"""
    3. Create the CVXPY problem
"""

# i) Variables

x = cp.Variable(3)
cons = [A@x >=0]


# ii) Objective

obj_fun = cp.norm(A@x-data,p=2)
opt_prob=cp.Problem(cp.Minimize(obj_fun),cons)



"""
    4. Assemble the solution
"""


# i) Solve problem

opt_prob.solve(verbose=True)


# ii) Assemble solution

x_opt_value = x.value
data_hat = A@x_opt_value



"""
    5. plots and illustrations  
"""

# i) Data

plt.figure(1, dpi=300)
sns.scatterplot(x=time,y=data.flatten()).set(title = 'data and model fit', xlabel = 'time', ylabel = 'observation = concenttrations')
sns.lineplot(x=time, y=data_hat, label = 'with constraints')
sns.lineplot(x=time, y=data_hat_ls, label = 'least squares')
plt.legend()



































