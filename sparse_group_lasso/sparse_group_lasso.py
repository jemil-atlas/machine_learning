"""
The goal of this script is to train a sparse group lasso using cvxpy.
For this, do the following
    1. Definitions and imports
    2. Generate some data
    3. Construct optimization problem
    4. Optimize and assemble solution
    5. Summarize and plot results
    
Author: Dr. Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.ch.
Copyright: 2021-2022, Atlas optimization GmbH, Zurich, Switzerland. All rights reserved.

"""


"""
    1. Definitions and imports -----------------------------------------------
"""


# i) Imports

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


# ii) Definitions

np.random.seed(0)

n_sig = 3     # Nr signature dimensions
n_sc = 33     # Nr spectral channels
n_feat = n_sig*n_sc   # Nr of features     
n_data_class = 100   # Nr data points per class
n_data = n_data_class*2


"""
    2. Generate some data ----------------------------------------------------
"""

# i) Multivariate normal observation setup

mu_1 = 1*np.ones([n_feat])    # mean class 1
mu_2 = -1*np.ones([n_feat])   # mean class 2

sigma = 0.5*np.eye(n_feat)

data_mat_1 = np.zeros([n_data_class,n_feat])
data_mat_2 = np.zeros([n_data_class,n_feat])


# ii) Simulate and assemble

for k in range(n_data_class):
    data_class_1 = np.random.multivariate_normal(mu_1, sigma)
    data_mat_1[k,:] = data_class_1

for k in range(n_data_class):
    data_class_2 = np.random.multivariate_normal(mu_2, sigma)
    data_mat_2[k,:] = data_class_2

data = np.vstack((data_mat_1,data_mat_2))


# iii) Class labels

labels = np.hstack((-np.ones(n_data_class),np.ones(n_data_class)))
labels = np.reshape(labels,[n_data,1])



"""
    3. Construct optimization problem ----------------------------------------
"""


# i) Give typical names

y = labels
X = data.T


# ii) Invoke variables

beta_0 = cp.Variable(shape=1)
beta = cp.Variable(shape=[n_feat,1])
z = cp.Variable(shape=[n_data,1])
z_p = cp.Variable(shape=[n_data,1],nonneg=True)
z_m = cp.Variable(shape=[n_data,1],nonneg=True)

B=cp.reshape(beta, [n_sc,n_sig])


# iii) Constraints

cons = []
cons = cons+[z == z_p-z_m]
for k in range(n_data):
    cons = cons+[z[k] == 1-y[k]@(beta_0+X[:,k]@beta)]
    

# iv) Create problem


lumda=350
obj_fun = np.ones([n_data,1]).T@z_p + lumda*cp.mixed_norm(B,2,1)
opt_prob=cp.Problem(cp.Minimize(obj_fun),constraints=cons)



"""
    4. Optimize and assemble solution ----------------------------------------
"""


# i) Solve and assemble

opt_prob.solve(verbose=True)

B_opt_value=B.value
beta_opt_value=beta.value
beta_0_opt_value=beta_0.value


# ii) Assemble classifier

classify = lambda x: np.sign(beta_0_opt_value+x.T@beta_opt_value)




"""
    5. Summarize and plot results --------------------------------------------
"""


# i) Test classifier on synthetic data

y_hat=np.zeros([n_data])
for k in range(n_data):
    y_hat[k]=classify(X[:,k])

# ii) Plots

plt.figure(1,dpi=300)
plt.scatter(np.linspace(0,1,n_data),y_hat-y.flatten())
plt.xlabel('datapoint')
plt.ylabel('error')
plt.title('Error classification')


plt.figure(2,dpi=300)
plt.imshow(B_opt_value)
plt.title('Coefficient distribution')
plt.xlabel('Signature dimension')
plt.ylabel('Spectral channel')

















