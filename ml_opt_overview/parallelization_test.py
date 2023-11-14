#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to test the gains from parallelization via the multi-
processing toolbox
For this, do the following:
    1. Definitions and imports
    2. Define compute task
    3. Compute and evaluate

"""

"""
    1. Definitions and imports
"""


from multiprocessing import Pool
import time
import numpy as np



"""
    2. Define compute task
"""


# Define a computationally intensive function
def compute(n):
    return sum(np.sin(i) * np.cos(i) for i in range(n))

# Create an array of tasks
tasks = [10**5] * 16



"""
    3. Compute and evaluate
"""

# Unparallelized execution
start_time = time.time()
unparallelized_results = list(map(compute, tasks))
unparallelized_time = time.time() - start_time

# Parallelized execution using 4 worker processes
start_time = time.time()
with Pool(processes=4) as pool:
    parallelized_results = pool.map(compute, tasks)
parallelized_time = time.time() - start_time

print("unparallelized time : {}     parallelized time : {}".format(unparallelized_time, parallelized_time))