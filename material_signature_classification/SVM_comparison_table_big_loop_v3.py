"""
The goal of this script is to catalogue the results of applying svm to datasets 
of decreasing information. The loss in classification accuracy is to be 
illustrated.

For this, do the following:
    1. Imports and definitions
    2. Prepare loop
    3. Set up data for SVM
    4. Train and test SVM
    5. Plots and illustrations
"""


"""
    1. Imports and definitions -----------------------------------------------
"""


# i) Imports

import numpy as np
import matplotlib.pyplot as plt
import copy as copy

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# ii) Load data 

Power_10p=np.load('Material_classification_python/Power_10p.npy')
Phase_10p=np.load('Material_classification_python/Phase_10p.npy')
Polarization_10p=np.load('Material_classification_python/Polarization_10p.npy')
wavelength_vector=np.load('Material_classification_python/wavelength_vector.npy')

Power_10p=np.delete(Power_10p,[1,2],0)
Phase_10p=np.delete(Phase_10p,[1,2],0)
Polarization_10p=np.delete(Polarization_10p,[1,2],0)

n_wavelengths=Phase_10p.shape[2]
n_datasets=Phase_10p.shape[1]
n_materials=3
n_simu=1000

table_switch=1              # 0 for broad band 1 for narrow band



"""
    2. Prepare loop ----------------------------------------------------------
"""


# i) Prepare cases for loop

# case_key describes which data is used: [Power, Phase, Polarization] where a 
# 0 indicates no use and a 1 does indicate use.  
index_tuple=np.where(np.zeros([2,2,2])==0)
case_key=np.vstack((index_tuple[0],index_tuple[1],index_tuple[2]))
case_key=np.fliplr(case_key)
case_key=np.delete(case_key,-1,1)

n_cases=case_key.shape[1]


# ii) Prepare wavelength vectors for loop

# Decreasing number of wavelengths: first in steps of 3for narrower bands,
# then one at a time for roader bands
if table_switch==0:
    n_wavelength_sequence=np.array([7,4,2,1])
    n_cases_wavelengths=n_wavelength_sequence.shape[0]
    sample_list=[np.array([0]),np.array([0,6]),np.array([0,3,6]), np.array([0,1,3,5,6]), np.array([0,1,2,3,4,5,6])]
    sample_list.reverse()
elif table_switch==1:
    n_wavelength_sequence=np.array([33,17,9,5,3,1])
    n_cases_wavelengths=n_wavelength_sequence.shape[0]
    sample_list=[np.array([16]),np.array([0,16,32]),np.array([0,8,16,24,32]), np.linspace(0,32,9).astype(int), np.linspace(0,32,17).astype(int), np.linspace(0,32,33).astype(int)]
    sample_list.reverse()

n_dim_data=n_wavelength_sequence[0]

# Create sequences of indices to sample at
to_del_list=[]

for k in range(n_cases_wavelengths):
    if table_switch==0:
        full_index_list=np.linspace(0,6,7).astype(int)
        temp_del_index=np.array(list(set(full_index_list)-set(sample_list[k])))
    elif table_switch==1:
        full_index_list=np.linspace(0,32,33).astype(int)
        temp_del_index=np.array(list(set(full_index_list)-set(sample_list[k])))
        
    to_del_list.append(temp_del_index)


classification_error_matrix=np.zeros([n_cases,n_cases_wavelengths, n_simu])
confusion_matrix_collection=np.zeros([n_cases,n_cases_wavelengths, n_simu,n_materials,n_materials])


# iv) Start up loop
for q in range(n_simu):
    print('Iteration', q)    
    for k in range(n_cases):
        case_index=case_key[:,k]
        print(case_index)
        
        # v) Create dataset in loop
        if table_switch==0:    
            data_x=np.concatenate((Power_10p[:,:,0:7],Phase_10p[:,:,0:7],Polarization_10p[:,:,0:7]), axis=2)
        elif table_switch==1:
            data_x=np.concatenate((Power_10p[:,:,7:],Phase_10p[:,:,7:],Polarization_10p[:,:,7:]), axis=2)
            
        data_y=np.kron(np.array([1,2,3,4,5]),np.ones([n_datasets]))
        
        train_data_x, test_data_x= train_test_split(np.transpose(data_x,[1,0,2]), test_size=0.3)
        n_train=train_data_x.shape[0]
        n_test=test_data_x.shape[0]
        
        train_data_y=np.kron(np.array([1,4,5]),np.ones([n_train]))
        test_data_y=np.kron(np.array([1, 4,5]),np.ones([n_test]))
        
        train_data_x=np.reshape(np.transpose(train_data_x,[1,0,2]),[n_train*n_materials, 3, n_dim_data])
        test_data_x=np.reshape(np.transpose(test_data_x,[1,0,2]),[n_test*n_materials, 3, n_dim_data])
        
    
    
        """
            3. Set up data for SVM ---------------------------------------------------
        """
                
                                            
    
        for m in range(n_cases_wavelengths):
            
            
            # i) Subsample data according to wavelengths
            
            ind_to_delete=to_del_list[m]
            train_data_x_wl=copy.copy(train_data_x)
            test_data_x_wl=copy.copy(test_data_x)
            if m==0:
                pass
            else:
                train_data_x_wl=np.delete(train_data_x_wl,ind_to_delete,2)
                test_data_x_wl=np.delete(test_data_x_wl,ind_to_delete,2)
            
            
            
            # ii) Subsample data according to cases
        
            train_data_x_loop=np.empty([n_train*n_materials,0])
            test_data_x_loop=np.empty([n_test*n_materials,0])
            for l in range(3):
                if case_index[l]==1:
                    train_data_x_loop=np.append(train_data_x_loop, train_data_x_wl[:,l,:],1)
                    test_data_x_loop=np.append(test_data_x_loop, test_data_x_wl[:,l,:],1)
                    
            n_dim_data_loop=train_data_x_loop.shape[1]
    
    
    
            """
                4. Train and test SVM --------------------------------------------------
            """
            
            
            # i) Linear SVM on whitened
            
            #  construct and train
            lin_svm = svm.LinearSVC()
            lin_svm.fit(train_data_x_loop,train_data_y)
            
            # evaluate on train and test
            lin_svm_pred_y_train=np.zeros([n_train*n_materials,1])
            lin_svm_pred_y_test=np.zeros([n_test*n_materials,1])
            
            for l in range(n_train*n_materials):
                lin_svm_pred_y_train[l]=lin_svm.predict(np.reshape(train_data_x_loop[l,:],[1,n_dim_data_loop]))
            
            for l in range(n_test*n_materials):
                lin_svm_pred_y_test[l]=lin_svm.predict(np.reshape(test_data_x_loop[l,:],[1,n_dim_data_loop]))
            
            # lin_svm_pred_y=np.vstack((lin_svm_pred_y_train,lin_svm_pred_y_test))
            # lin_svm_true_y=np.vstack((np.reshape(train_data_y,[n_train*n_materials,1]),np.reshape(test_data_y,[n_test*n_materials,1])))
            # lin_svm_error_vec=np.abs(lin_svm_pred_y-lin_svm_true_y)
            # lin_svm_error_vec[lin_svm_error_vec>=1]=1
            
            lin_svm_test_error_vec=np.abs(np.reshape(test_data_y,[n_test*n_materials,1])-lin_svm_pred_y_test)     
                      
            lin_svm_test_error_indication=1*(lin_svm_test_error_vec>=1)
            classification_error_matrix[k,m,q]=np.mean(lin_svm_test_error_indication) 
            confusion_matrix_collection[k,m,q,:,:]=confusion_matrix(np.reshape(test_data_y,[n_test*n_materials,1]),lin_svm_pred_y_test,normalize='true')
            # print(np.sum(lin_svm_error_vec))
    
    



"""
    5. Plots and illustrations ------------------------------------------------
"""


# i) Illustrate matrix

classification_error_matrix_mean=np.mean(classification_error_matrix,2)
confusion_matrix_mean_collection=np.mean(confusion_matrix_collection,2)

fig=plt.figure(dpi=500)
im=plt.imshow(classification_error_matrix_mean, vmin=0, vmax=1)
plt.colorbar(im)
plt.xticks(np.arange(0, n_cases_wavelengths, step=1),n_wavelength_sequence )
plt.yticks(np.arange(0, 7, step=1), case_key.T )
plt.xlabel('Nr of Wavelengths')
plt.ylabel('Case configuration')
plt.title('Ratio of errors for different configurations')


fig=plt.figure(dpi=500)
im=plt.imshow(confusion_matrix_mean_collection[3,0,:,:], vmin=0, vmax=1)
plt.colorbar(im)
plt.xticks(np.arange(0, 3, step=1),[1,4,5])
plt.yticks(np.arange(0, 3, step=1),[1,4,5])
plt.xlabel('Material class predicted')
plt.ylabel('Material class true')
plt.title('Confusion matrix Power all wl')

fig=plt.figure(dpi=500)
im=plt.imshow(confusion_matrix_mean_collection[6,0,:,:], vmin=0, vmax=1)
plt.colorbar(im)
plt.xticks(np.arange(0, 3, step=1),[1,4,5])
plt.yticks(np.arange(0, 3, step=1),[1,4,5])
plt.xlabel('Material class predicted')
plt.ylabel('Material class true')
plt.title('Confusion matrix Polarization all wl')











