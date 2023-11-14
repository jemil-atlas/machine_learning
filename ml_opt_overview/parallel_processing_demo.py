#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script demonstrates basic usage of python multiprocessing tools.
        
        
Author: Dr. Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.ch.
Copyright: 2021-2022, Atlas optimization GmbH, Zurich, Switzerland. All rights reserved.

"""


import numpy as np
from timeit import timeit

# Prepare data
np.random.RandomState(100)
arr = np.random.randint(0, 10, size=[200000, 5])
data = arr.tolist()
data[:5]



def howmany_within_range(row, minimum, maximum):
    """Returns how many numbers lie within `maximum` and `minimum` in a given `row`"""
    count = 0
    for n in row:
        if minimum <= n <= maximum:
            count = count + 1
    return count

t0 = timeit()
results = []
for row in data:
    results.append(howmany_within_range(row, minimum=4, maximum=8))
t1 = timeit()
print('The elapsed time is :', t1-t0)
#> [3, 1, 4, 4, 4, 2, 1, 1, 3, 3]




import multiprocessing as mp

# Step 1: Init multiprocessing.Pool()
pool = mp.Pool(mp.cpu_count())

# Step 2: `pool.apply` the `howmany_within_range()`
t0 = timeit()
results = [pool.apply(howmany_within_range, args=(row, 4, 8)) for row in data]
t1 = timeit()

print('The elapsed time with mp is :', t1-t0)
# Step 3: Don't forget to close
pool.close()    
