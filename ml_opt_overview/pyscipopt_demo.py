#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script demonstrates basic usage of pyscipopt python scip interface. Examples
taken from https://scipbook.readthedocs.io/en/latest/intro.html .
        
        
Author: Dr. Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.ch.
Copyright: 2021-2022, Atlas optimization GmbH, Zurich, Switzerland. All rights reserved.
"""

from pyscipopt import Model

# Problem 1: LP
model = Model("Simple linear optimization")

x1 = model.addVar(vtype="C", name="x1")
x2 = model.addVar(vtype="C", name="x2")
x3 = model.addVar(vtype="C", name="x3")

model.addCons(2*x1 + x2 + x3 <= 60)
model.addCons(x1 + 2*x2 + x3 <= 60)
model.addCons(x3 <= 30)

model.setObjective(15*x1 + 18*x2 + 30*x3, "maximize")

model.optimize()

if model.getStatus() == "optimal":
    print("Optimal value:", model.getObjVal())
    print("Solution:")
    print("  x1 = ", model.getVal(x1))
    print("  x2 = ", model.getVal(x2))
    print("  x3 = ", model.getVal(x3))
else:
    print("Problem could not be solved to optimality")

# Problem 2: MILP
model = Model("Simple mixed integer linear optimization")

x = model.addVar(vtype = "I", name = "x", lb = 0)
y = model.addVar(vtype = "I", name = "y", lb = 0)
z = model.addVar(vtype = "I", name = "z", lb = 0)

model.addCons(2*x + 4*y + 8*z == 80)
model.addCons(x + y + z == 32)

model.setObjective(y + z, "minimize")
model.optimize()

print("Optimal value:", model.getObjVal())
print("Solution:")
print("  x = ", model.getVal(x))
print("  y = ", model.getVal(y))
print("  z = ", model.getVal(z))


# Problem 3: MILP with dictvalued var
model = Model("Simple MILP with dict")

x={}
for k in range(3):
    x[k] = model.addVar(vtype = "I", name = "x_{}".format(k), lb = 0)

model.addCons(2*x[0] + 4*x[1] + 8*x[2] == 80)
model.addCons(x[0] + x[1] + x[2] == 32)

model.setObjective(x[1] + x[2], "minimize")
model.optimize()

print("Optimal value:", model.getObjVal())
print("Solution:")
for k in range(3):
    print("  x[{}] = ".format(k), model.getVal(x[k]))



















