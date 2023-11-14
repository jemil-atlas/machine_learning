#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to showcase optimization with cvxpy. To this end, a 
the simple problem of fitting a line via least squares is formulated via cvxpy.
For this, do the following:
        1. Definitions and imports
        2. Generate Data 
        3. Define Problem
        4. Solve problem and assemble solution

Author: Dr. Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.ch.
Copyright: 2021-2022, Atlas optimization GmbH, Zurich, Switzerland. All rights reserved.

"""

"""
        1. Definitions and imports -------------------------------------------
"""


# i) Imports

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.stats import norm

# ii) Definitions

n = 100
t = np.linspace(0,1,n)
n_sample = 5
t_sample = np.linspace(0,1,n_sample)



"""
        2. Generate Data ------------------------------------------------------
"""


# i) Data

y = np.array([-0.1, 0.3, 0.55, 0.70, 1.4]).reshape([n_sample,1])
A = np.hstack((np.ones([n_sample,1]), np.reshape(t_sample,[n_sample,1])))
A_full = np.hstack((np.ones([n,1]), np.reshape(t,[n,1])))



"""
        2. Define problem ------------------------------------------------------
"""


# i) Variables

lumbda = cp.Variable(shape = [2,1])


# ii) objective function

obj_fun = cp.norm(A@lumbda - y,2)


# iii) constraints

cons = []
cons = cons + [lumbda[0]>=0] # Condition: abscissa >=0 (initial happiness)
cons = cons + [A_full@lumbda >= 0]    # Condition: Happiness values normalized between 0,1
cons = cons + [A_full@lumbda <= 1]

# Prob constraint : p(g^Tlumbda >=0) >=0.95 
g_bar = np.array([[1],[-1]]) # abscissa bigger than slope
Sigma_sqrt = 0.1*np.eye(2)
cons = cons + [norm.ppf(0.95)*cp.norm(Sigma_sqrt@lumbda) - g_bar.T@lumbda<=0] # implied SOCC


# iv) Problem

opt_prob = cp.Problem(cp.Minimize(obj_fun),constraints = cons)



"""
        4. Solve problem and assemble solution --------------------------------
"""


# i) Solve and extract

opt_prob.solve(verbose=True)
lumbda_val = lumbda.value


# ii) Illustrate data & solution

fig = plt.figure(1,dpi=300)
plt.scatter(t_sample,y)
plt.title('Data')
plt.xlabel('Money')
plt.ylabel('Happiness')


fig = plt.figure(2,dpi=300)
plt.scatter(t_sample,y)
plt.plot(t,A_full@lumbda_val)
plt.title('Data')
plt.xlabel('Money')
plt.ylabel('Happiness')
