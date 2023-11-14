#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script demonstrates basic usage of pyscipopt python scip interface. Examples
taken from https://scipbook.readthedocs.io/en/latest/intro.html .
        
        
Author: Dr. Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.ch.
Copyright: 2021-2022, Atlas optimization GmbH, Zurich, Switzerland. All rights reserved.
"""


# # First, with cvxpy
# import numpy as np
# import cvxpy as cp

# a_bar = 2*np.ones(2)
# Sigma = np.eye(2)
# c = np.ones(2)
# b = 1

# x = cp.Variable(2)
# obj_fun = c.T@x
# cons = []
# cons = cons +[-a_bar.T@x + cp.norm(Sigma@x,2) <= -1] # a^Tx >=b

# prob = cp.Problem(cp.Minimize(obj_fun),constraints = cons)
# prob.solve(verbose = True)
# x_val = x.value
# print( 'x = ', x_val)

# Then, with pyscipopt
import numpy as np
from pyscipopt import Model, quicksum

a_bar = 2*np.ones(2)
Sigma = np.eye(2)
c = np.ones(2)
b = 1

# Problem 1: SOCP robust LP
model = Model("Lp with robustness constraint")
x = {}
x[0] = model.addVar(vtype="C", name="x1")
x[1] = model.addVar(vtype="C", name="x2")

model.addCons(quicksum(-a_bar[k]*x[k] for k in range(2)) + (quicksum(Sigma[k,k]*x[k]*x[k] for k in range(2)))**(1/2) <= -b)
# model.addCons(quicksum(np.abs(x[k]) for k in range(2)) ==1)
model.addCons(np.sin(x[0]) == 0.5)  #< ----no

model.setObjective(x[0]+x[1], "minimize")

model.optimize()

if model.getStatus() == "optimal":
    print("Optimal value:", model.getObjVal())
    print("Solution:")
    print("  x1 = ", model.getVal(x[0]))
    print("  x2 = ", model.getVal(x[1]))
else:
    print("Problem could not be solved to optimality")













