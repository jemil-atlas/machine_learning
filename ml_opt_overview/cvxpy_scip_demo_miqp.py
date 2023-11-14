#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 23:17:00 2023

@author: jemil
"""


import cvxpy as cp
import numpy as np


# Generate a random problem
np.random.seed(0)
m, n= 40, 25

A = np.random.rand(m, n)
b = np.random.randn(m)

# Construct a CVXPY problem
x = cp.Variable(n, integer=True)
objective = cp.Minimize(cp.sum_squares(A @ x - b))
prob = cp.Problem(objective)
prob.solve(solver = 'SCIP')

print("Status: ", prob.status)
print("The optimal value is", prob.value)
print("A solution x is")
print(x.value)



