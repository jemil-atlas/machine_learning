#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides functionality for showcasing code and functionality of sklearn,
a classic python library for machine learning. This is done by fitting some svc
to radial data. 
For this, do the following:
    1. Imports and definitions
    2. Generate some data
    3. Solve the sklearn problem
    4. Plots and illustrations        
    
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
from sklearn import svm


# ii) Definitions

n_sample = 100
np.random.seed(1)

n_disc = 50
x = np.linspace(-2,2,n_disc)
xx,yy = np.meshgrid(x,x)



"""
    2. Generate some data
"""


# i) True model

true_dr = lambda x: np.sign(1*(x.T@x >=1) -0.5)

data_x = np.random.normal(0,1,[2,n_sample])
data_label = np.diag(true_dr(data_x))

data_plus = data_x.T[data_label>=0]
data_minus = data_x.T[data_label<=0]



"""
    3. Solve the sklearn problem
"""


# i) Invoke & fit svm

rbf_svm = svm.SVC(kernel = 'rbf')
rbf_svm.fit(data_x.T, data_label)


# ii) Assemble solution

decision_image = np.zeros([n_disc,n_disc])

for k in range(n_disc):
    for l in range(n_disc):
        decision_image[k,l] = rbf_svm.predict(np.array([[xx[k,l], yy[k,l]]]))




"""
    4. Plots and illustrations  
"""


# i) Data

plt.figure(1, dpi=300)
sns.scatterplot(x= data_plus[:,0],y=data_plus[:,1], color = 'b', label = 'class 1').set(title = 'data and classification', xlabel = 'feature_1', ylabel = 'feature_2')
sns.scatterplot(x= data_minus[:,0],y=data_minus[:,1], color = 'r', label = 'class 2').set(title = 'data and classification', xlabel = 'feature_1', ylabel = 'feature_2')
plt.legend()


# ii) Decision boundary

plt.imshow(decision_image, extent = [-2,2, -2,2])
































