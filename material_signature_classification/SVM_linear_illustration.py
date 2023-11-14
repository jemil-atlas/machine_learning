"""
The goal of this script is to try out a linear svm.
For this, do the following:
    1. Imports and definitions
    2. Preprocessing
    3. SVM: Linear, whitened
    4. SVM: Linear, unwhitened
"""


"""
    1. Imports and definitions -----------------------------------------------
"""


# i) Imports

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.model_selection import train_test_split


# ii) Load data 

Phase_10p=np.load('Material_classification_python/Phase_10p.npy')
Power_10p=np.load('Material_classification_python/Power_10p.npy')
Polarization_10p=np.load('Material_classification_python/Polarization_10p.npy')

n_wavelengths=Phase_10p.shape[2]
n_datasets=Phase_10p.shape[1]
n_dim_data=3*n_wavelengths
n_materials=5


# iii) Make into one dataset, concatenate Reflectance and phase of 10p observations

data_x=np.concatenate((Power_10p,Phase_10p,Polarization_10p), axis=2)
data_y=np.kron(np.array([1,2,3,4,5]),np.ones([n_datasets]))

train_data_x, test_data_x= train_test_split(np.transpose(data_x,[1,0,2]), test_size=0.3)
n_train=train_data_x.shape[0]
n_test=test_data_x.shape[0]

train_data_y=np.kron(np.array([1,2,3,4,5]),np.ones([n_train]))
test_data_y=np.kron(np.array([1,2,3,4,5]),np.ones([n_test]))

train_data_x=np.reshape(np.transpose(train_data_x,[1,0,2]),[n_train*n_materials,n_dim_data])
test_data_x=np.reshape(np.transpose(test_data_x,[1,0,2]),[n_test*n_materials,n_dim_data])



"""
    2. Preprocessing ----------------------------------------------------------
"""


# i) Create mean vector and empirical covariance matrix

mean_full=np.reshape(np.mean(train_data_x,0),[1,n_dim_data])
covmat_full=(1/n_train)*(train_data_x.T@train_data_x-mean_full.T@mean_full)

Sigma_normalization=np.linalg.pinv(np.diag(np.sqrt(np.diag(covmat_full))))

# ii) Whiten the data

train_data_x_whitened=(train_data_x.T-np.repeat(mean_full.T,n_train*n_materials,1)).T
test_data_x_whitened=(test_data_x.T-np.repeat(mean_full.T,n_test*n_materials,1)).T

train_data_x_whitened=(Sigma_normalization@train_data_x.T).T
test_data_x_whitened=(Sigma_normalization@test_data_x.T).T



"""
    3. SVM: Linear, whitened --------------------------------------------------
"""


# i) Linear SVM on whitened

# #  construct and train
# lin_svm = svm.LinearSVC()
# lin_svm.fit(train_data_x_whitened,train_data_y)

# # evaluate on train and test
# lin_svm_pred_y_train=np.zeros([n_train*n_materials,1])
# lin_svm_pred_y_test=np.zeros([n_test*n_materials,1])

# for k in range(n_train*n_materials):
#     lin_svm_pred_y_train[k]=lin_svm.predict(np.reshape(train_data_x_whitened[k,:],[1,n_dim_data]))

# for k in range(n_test*n_materials):
#     lin_svm_pred_y_test[k]=lin_svm.predict(np.reshape(test_data_x_whitened[k,:],[1,n_dim_data]))

# lin_svm_pred_y=np.vstack((lin_svm_pred_y_train,lin_svm_pred_y_test))
# lin_svm_true_y=np.vstack((np.reshape(train_data_y,[n_train*n_materials,1]),np.reshape(test_data_y,[n_test*n_materials,1])))
# lin_svm_error_vec=np.abs(lin_svm_pred_y-lin_svm_true_y)
# lin_svm_error_vec[lin_svm_error_vec>=1]=1

# print(np.sum(lin_svm_error_vec))



"""
    4. SVM: Linear, unwhitened ------------------------------------------------
"""


# i) Linear SVM on unwhitened

# construct and train
lin_svm = svm.LinearSVC()
lin_svm.fit(train_data_x,train_data_y)

# evaluate on train and test
lin_svm_pred_y_train=np.zeros([n_train*n_materials,1])
lin_svm_pred_y_test=np.zeros([n_test*n_materials,1])

for k in range(n_train*n_materials):
    lin_svm_pred_y_train[k]=lin_svm.predict(np.reshape(train_data_x[k,:],[1,n_dim_data]))

for k in range(n_test*n_materials):
    lin_svm_pred_y_test[k]=lin_svm.predict(np.reshape(test_data_x[k,:],[1,n_dim_data]))

lin_svm_pred_y=np.vstack((lin_svm_pred_y_train,lin_svm_pred_y_test))
lin_svm_true_y=np.vstack((np.reshape(train_data_y,[n_train*n_materials,1]),np.reshape(test_data_y,[n_test*n_materials,1])))
lin_svm_error_vec=np.abs(lin_svm_pred_y-lin_svm_true_y)
lin_svm_error_vec[lin_svm_error_vec>=1]=1

print(np.sum(lin_svm_error_vec))




