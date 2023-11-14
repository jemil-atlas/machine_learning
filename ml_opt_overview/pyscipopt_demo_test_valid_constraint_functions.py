#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script demonstrates basic usage of pyscipopt python scip interface. The goal
is to find out which type of functions can be used to formulate constraints and
which are disallowed.
        
        
Author: Dr. Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.ch.
Copyright: 2021-2022, Atlas optimization GmbH, Zurich, Switzerland. All rights reserved.
"""


"""
    1. Imports and definitions -----------------------------------------------
"""


# i) Imports

import numpy as np
from pyscipopt import Model
from pyscipopt import quicksum, quickprod, log, exp, sqrt #sin, cos <- not implemented yet
import matplotlib.pyplot as plt


# ii) Definitions

n=10


"""
    2. Construct model and test constraints ----------------------------------
"""

# i) Invoke model

model = Model("Nonlinear constraints")

x = {}
for k in range(n):
    x[k] = model.addVar(vtype = "C", name = "x_{}".format(k), lb = 0)

model.setObjective(quicksum(x[k] for k in range(n)), "minimize")


# ii) Try out different constraints

# EQUALITIES

# UNIVARIATE
# quadratic constraint
# model.addCons(x[0]**2 == 2)

# # sqrt constraint
# model.addCons(x[0]**(1/2) == 2)

# # sqrt constraint reformulated
# model.addCons(sqrt(x[0]) == 2)

# # other power constraint
# model.addCons(x[0]**(4/7) == 2)

# # abs value constraint
# model.addCons(np.abs(x[0]) == 2)                # <------ interestingly, this does work?

# # exp constraint
# model.addCons(exp(x[0]) == 2)

# # log constraint
# model.addCons(log(x[0]) == 2)




# MULTIVARIATE

# # sqrt constraint
# model.addCons((x[0] + x[1])**(1/2) == 2)

# # sqrt constraint reformulated
# model.addCons(sqrt(x[0]*x[1]) == 2)

# # division constraint
# model.addCons(x[0]/x[1]== 2)

# # other power constraint
# model.addCons(x[0]**x[1] == 2)              # <----- does not work

# # abs value constraint
# model.addCons(np.abs(x[0])*np.abs(x[1]) == 2)                 # <------ interestingly, this does work?

# # exp constraint 1
# model.addCons(exp(x[0]*x[1]) == 2)

# # exp constraint 2
# model.addCons(exp(x[0]*x[1])*x[2] == 2)

# # exp constraint 3
# model.addCons(exp(x[0]*exp(x[1])) == 2)

# # log constraint
# model.addCons(log(x[0]) == 2)



# INEQUALITIES

# UNIVARIATE

# MULTIVARIATE

# iii) Solve problem

model.optimize()



"""
    3. Assemble solutions and illustrate -------------------------------------
"""


# i) Assemble solution

x_vec = np.zeros(n)
for k in range(n):
    x_vec[k] = model.getVal(x[k])


# ii) Print and plot x
plt.plot(x_vec)
print('x at location 0 and 1 are ', x_vec[0], x_vec[1])
