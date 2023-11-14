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
n_classes = 4     # Nr of classes
n_feat = n_sig*n_sc   # Nr of features     
n_data_class = 10   # Nr data points per class
n_data = n_data_class*n_classes


"""
    2. Generate some data ----------------------------------------------------
"""

# i) Multivariate normal observation setup

mu=np.zeros([n_feat,n_classes])  # mu matrix contains vectors of expected values in columns

for k in range(n_classes):
    mu[:,k] = k*np.ones(n_feat)


sigma = 0.1*np.eye(n_feat)

data_mat = np.zeros([n_data,n_feat])


# ii) Simulate and assemble

for k in range(n_classes):
    for l in range(n_data_class):

        data_class_temp = np.random.multivariate_normal(mu[:,k], sigma)
        data_mat[l+n_data_class*(k),:] = data_class_temp


data = data_mat


# iii) Class labels

labels = np.zeros(n_data)

for k in range(n_classes):
    for l in range(n_data_class):
         labels[l+n_data_class*(k)] = k


"""
    3. Construct optimization problem ----------------------------------------
"""


# i) Give typical names

y = labels
X = data.T


# ii) Invoke variables

b_vec = cp.Variable(shape=n_classes)
w_mat = cp.Variable(shape=[n_sc,n_sig*n_classes])

xi = cp.Variable(shape=[n_data,n_classes],nonneg=True)

# w_k = cp.vec(w_mat[k*n_sig:(k+1)*n_sig])

# iii) Constraints

cons = []

for k in range(n_classes):
    for l in range(n_data):
        # ind_y = (y[l]).astype(int)
        # cons = cons+[(cp.vec(w_mat[:,ind_y:ind_y+n_sig])-cp.vec(w_mat[:,k:k+n_sig])).T@X[:,l]+b_vec[ind_y]-b_vec[k]>=1-xi[l,k]]
        if y[l] == k:
            pass
        else:
            ind_y = (y[l]).astype(int)
            # cons = cons+[(cp.vec(w_mat[:,ind_y:ind_y+n_sig])-cp.vec(w_mat[:,k:k+n_sig])).T@X[:,l]+b_vec[ind_y]-b_vec[k]>=1-xi[l,k]]
            cons = cons+[(cp.vec(w_mat[:,ind_y*n_sig:(ind_y+1)*n_sig])-cp.vec(w_mat[:,k*n_sig:(k+1)*n_sig])).T@X[:,l]+b_vec[ind_y]-b_vec[k]>=1-xi[l,k]]

# iv) Create problem


lumda=100
obj_fun = lumda*cp.mixed_norm(w_mat,2,1) + np.ones(n_data*n_classes).T@ cp.vec(xi)
opt_prob=cp.Problem(cp.Minimize(obj_fun),constraints=cons)



"""
    4. Optimize and assemble solution ----------------------------------------
"""


# i) Solve and assemble

opt_prob.solve(verbose=True)

w_mat_opt=w_mat.value
b_vec_opt=b_vec.value
xi_opt=xi.value

w_vecs_opt = np.zeros([n_feat,n_classes]) # each column = weight vectors for one class
for k in range(n_classes):
    w_vecs_opt[:,k]=(w_mat_opt[:,k*n_sig:(k+1)*n_sig]).flatten(order = 'F')


# ii) Assemble classifier

def classify(x):
    score = np.zeros(n_classes)
    for k in range(n_classes):
        score[k] = (w_vecs_opt[:,k].T@x+b_vec_opt[k])
        
    return np.argmax(score)
    


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
plt.imshow(w_mat_opt)
plt.colorbar()
plt.title('Coefficient distribution')
plt.xlabel('Signature dimension x classes')
plt.ylabel('Spectral channel')

















