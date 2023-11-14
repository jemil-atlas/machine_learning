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
n_sample = 100
t_sample = np.linspace(0,1,n_sample)
t_c = 0.5

t_1 = t[t<=t_c]
t_2 = t[t>t_c]
n_1 = len(t_1)
n_2 = len(t_2)


# iii) Seeding & sampling

np.random.seed(1)

t_sample_1 = t_sample[t_sample<=t_c]
t_sample_2 = t_sample[t_sample>t_c]
n_sample_1 = len(t_sample_1)
n_sample_2 = len(t_sample_2)



"""
        2. Generate Data ------------------------------------------------------
"""


# i) Data
sigma_noise= 0.1
random_intersection = np.random.normal(0,1)
line_1 = np.linspace(np.random.normal(0,1),random_intersection,n_sample_1)
line_2 = np.linspace(random_intersection,np.random.normal(0,1),n_sample_2)

y_1 = np.random.multivariate_normal(line_1, sigma_noise*np.eye(n_sample_1)).reshape([n_sample_1,1])
y_2 = np.random.multivariate_normal(line_2, sigma_noise*np.eye(n_sample_2)).reshape([n_sample_2,1])
A_1 = np.hstack((np.ones([n_sample_1,1]), np.reshape(t_sample_1,[n_sample_1,1])))
A_1_full = np.hstack((np.ones([n_1,1]), np.reshape(t_1,[n_1,1])))
A_2 = np.hstack((np.ones([n_sample_2,1]), np.reshape(t_sample_2,[n_sample_2,1])))
A_2_full = np.hstack((np.ones([n_2,1]), np.reshape(t_2,[n_2,1])))



"""
        2. Define problem ------------------------------------------------------
"""


# i) Variables

lumbda_1 = cp.Variable(shape = [2,1])
lumbda_2 = cp.Variable(shape = [2,1])


# ii) objective function

obj_fun = cp.sum_squares(A_1@lumbda_1 - y_1) + cp.sum_squares(A_2@lumbda_2 - y_2) 


# iii) constraints

a_intersect = np.array([[1],[t_c]])

cons = []
cons = cons + [a_intersect.T@ lumbda_1 - a_intersect.T@lumbda_2 ==0] # Condition: lines intersect at t_c


# iv) Problem

opt_prob = cp.Problem(cp.Minimize(obj_fun),constraints = cons)



"""
        4. Solve problem and assemble solution --------------------------------
"""


# i) Solve and extract

opt_prob.solve(solver = 'SCIP', verbose=True)
lumbda_1_val = lumbda_1.value
lumbda_2_val = lumbda_2.value

# ii) Illustrate data & solution

fig = plt.figure(1,dpi=300)
plt.scatter(t_sample,np.vstack((y_1,y_2)))
plt.title('Data')
plt.xlabel('Money')
plt.ylabel('Happiness')


fig = plt.figure(2,dpi=300)
plt.scatter(np.hstack((t_sample_1,t_sample_2)),np.vstack((y_1,y_2)))
plt.plot(t,np.vstack((A_1_full@lumbda_1_val,A_2_full@lumbda_2_val)))
plt.title('Data')
plt.xlabel('Money')
plt.ylabel('Happiness')
