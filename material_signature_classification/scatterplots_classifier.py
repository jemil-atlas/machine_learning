"""
The goal of this script is to create some scatterplots that show different materials
clustering around different means.

For this, do the following:
    1. Imports and definitions
    2. Set up data 
    3. Plots and illustrations
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


# ii) Load data 

Power_10p=np.load('Material_classification_python/Power_10p.npy')
Phase_10p=np.load('Material_classification_python/Phase_10p.npy')
Polarization_10p=np.load('Material_classification_python/Polarization_10p.npy')
wavelength_vector=np.load('Material_classification_python/wavelength_vector.npy')

n_wavelengths=Phase_10p.shape[2]
n_datasets=Phase_10p.shape[1]
n_dim_data=n_wavelengths
n_materials=5



"""
    2. Set up data -----------------------------------------------------------
"""

# i) Set up configuration lambda: middle 1 lambda broad, narrow, 2 lambdas broad, narrow, 3 lambdas broad, narrow.

lambda_index_list=[np.array([3]), np.array([23]), np.array([2,5]), np.array([17,28]), np.array([0,3,6]), np.array([8,23,38])]
key_list_1_meas=[np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])]
key_list_2_meas=[np.array([1,1,0]),np.array([0,1,1]),np.array([1,0,1])]
key_list_3_meas=[np.array([1,1,1])]


# ii) Generate data associated to configurations

data_x=np.concatenate((Power_10p,Phase_10p,Polarization_10p), axis=2)
data_x_reshaped=np.reshape(data_x,[n_materials*n_datasets,3,n_wavelengths])

# 3D

# 111b: [1,1,1], lambda middle, broad
data_wl=data_x_reshaped[:,:,lambda_index_list[0][0]]
data_111b=data_wl[:,[0,1,2]]

# 111n: [1,1,1], lambda middle, narrow
data_wl=data_x_reshaped[:,:,lambda_index_list[1][0]]
data_111n=data_wl[:,[0,1,2]]


# 1003b: [1,0,0] lambda_1, lambda_2, lambda_3, broad
data_wl=data_x_reshaped[:,:,[lambda_index_list[4][0],lambda_index_list[4][1], lambda_index_list[4][2]]]
data_1003b=np.reshape(data_wl[:,[0],:],[n_materials*n_datasets,3])

# 1003n: [1,0,0] lambda_1, lambda_2, lambda_3, narrow
data_wl=data_x_reshaped[:,:,[lambda_index_list[5][0],lambda_index_list[5][1], lambda_index_list[5][2]]]
data_1003n=np.reshape(data_wl[:,[0],:],[n_materials*n_datasets,3])

# 0103b: [0,1,0] lambda_1, lambda_2, lambda_3, broad
data_wl=data_x_reshaped[:,:,[lambda_index_list[4][0],lambda_index_list[4][1], lambda_index_list[4][2]]]
data_0103b=np.reshape(data_wl[:,[1],:],[n_materials*n_datasets,3])

# 0103n: [0,1,0] lambda_1, lambda_2, lambda_3, narrow
data_wl=data_x_reshaped[:,:,[lambda_index_list[5][0],lambda_index_list[5][1], lambda_index_list[5][2]]]
data_0103n=np.reshape(data_wl[:,[1],:],[n_materials*n_datasets,3])

# 0013b: [0,0,1] lambda_1, lambda_2, lambda_3, broad
data_wl=data_x_reshaped[:,:,[lambda_index_list[4][0],lambda_index_list[4][1], lambda_index_list[4][2]]]
data_0013b=np.reshape(data_wl[:,[2],:],[n_materials*n_datasets,3])

# 0013n: [0,0,0] lambda_1, lambda_2, lambda_3, narrow
data_wl=data_x_reshaped[:,:,[lambda_index_list[5][0],lambda_index_list[5][1], lambda_index_list[5][2]]]
data_0013n=np.reshape(data_wl[:,[2],:],[n_materials*n_datasets,3])



# 2D

# 110b: [1,1,0], lambda middle, broad
data_wl=data_x_reshaped[:,:,lambda_index_list[0][0]]
data_110b=data_wl[:,[0,1]]

# 110n: [1,1,0], lambda middle, narrow
data_wl=data_x_reshaped[:,:,lambda_index_list[1][0]]
data_110n=data_wl[:,[0,1]]

# 101b: [1,1,0], lambda middle, broad
data_wl=data_x_reshaped[:,:,lambda_index_list[0][0]]
data_101b=data_wl[:,[0,2]]

# 101n: [1,1,0], lambda middle, narrow
data_wl=data_x_reshaped[:,:,lambda_index_list[1][0]]
data_101n=data_wl[:,[0,2]]

# 011b: [1,1,0], lambda middle, broad
data_wl=data_x_reshaped[:,:,lambda_index_list[0][0]]
data_011b=data_wl[:,[1,2]]

# 011n: [1,1,0], lambda middle, narrow
data_wl=data_x_reshaped[:,:,lambda_index_list[1][0]]
data_011n=data_wl[:,[1,2]]




# 100b: [1,0,0] lambda_1, lambda_2, broad
data_wl=data_x_reshaped[:,:,[lambda_index_list[2][0],lambda_index_list[2][1]]]
data_100b=np.reshape(data_wl[:,[0],:],[n_materials*n_datasets,2])

# 100n: [1,0,0] lambda_1, lambda_2, narrow
data_wl=data_x_reshaped[:,:,[lambda_index_list[3][0],lambda_index_list[3][1]]]
data_100n=np.reshape(data_wl[:,[0],:],[n_materials*n_datasets,2])

# 010b: [0,1,0] lambda_1, lambda_2, broad
data_wl=data_x_reshaped[:,:,[lambda_index_list[2][0],lambda_index_list[2][1]]]
data_010b=np.reshape(data_wl[:,[1],:],[n_materials*n_datasets,2])

# 010n: [0,1,0] lambda_1, lambda_2, narrow
data_wl=data_x_reshaped[:,:,[lambda_index_list[3][0],lambda_index_list[3][1]]]
data_010n=np.reshape(data_wl[:,[1],:],[n_materials*n_datasets,2])

# 001b: [0,0,1] lambda_1, lambda_2, broad
data_wl=data_x_reshaped[:,:,[lambda_index_list[2][0],lambda_index_list[2][1]]]
data_001b=np.reshape(data_wl[:,[2],:],[n_materials*n_datasets,2])

# 001n: [0,0,0] lambda_1, lambda_2, narrow
data_wl=data_x_reshaped[:,:,[lambda_index_list[3][0],lambda_index_list[3][1]]]
data_001n=np.reshape(data_wl[:,[2],:],[n_materials*n_datasets,2])



"""
    3. Plots and illustrations ------------------------------------------------
"""


# i) Illustrate features 3D Plots

# k = cardboard, r=foam,  b=plaster, g= plastic, c= woodboard


# 111b
fig=plt.figure(dpi=500)
ax = fig.add_subplot(projection='3d')
ax.scatter(data_111b[0:9,0], data_111b[0:9,1], data_111b[0:9,2],c='k')
ax.scatter(data_111b[10:19,0], data_111b[10:19,1], data_111b[10:19,2],c='r')
ax.scatter(data_111b[20:29,0], data_111b[20:29,1], data_111b[20:29,2],c='b')
ax.scatter(data_111b[30:39,0], data_111b[30:39,1], data_111b[30:39,2],c='g')
ax.scatter(data_111b[40:49,0], data_111b[40:49,1], data_111b[40:49,2],c='c')
ax.set_xlabel('Power')
ax.set_ylabel('Phase')
ax.set_zlabel('Polarization')
plt.title('Features 111b')
plt.show()


# 111n
fig=plt.figure(dpi=500)
ax = fig.add_subplot(projection='3d')
ax.scatter(data_111n[0:9,0], data_111n[0:9,1], data_111n[0:9,2],c='k')
ax.scatter(data_111n[10:19,0], data_111n[10:19,1], data_111n[10:19,2],c='r')
ax.scatter(data_111n[20:29,0], data_111n[20:29,1], data_111n[20:29,2],c='b')
ax.scatter(data_111n[30:39,0], data_111n[30:39,1], data_111n[30:39,2],c='g')
ax.scatter(data_111n[40:49,0], data_111n[40:49,1], data_111n[40:49,2],c='c')
ax.set_xlabel('Power')
ax.set_ylabel('Phase')
ax.set_zlabel('Polarization')
plt.title('Features 111n')
plt.show()


# 1003b
fig=plt.figure(dpi=500)
ax = fig.add_subplot(projection='3d')
ax.scatter(data_1003b[0:9,0], data_1003b[0:9,1], data_1003b[0:9,2],c='k')
ax.scatter(data_1003b[10:19,0], data_1003b[10:19,1], data_1003b[10:19,2],c='r')
ax.scatter(data_1003b[20:29,0], data_1003b[20:29,1], data_1003b[20:29,2],c='b')
ax.scatter(data_1003b[30:39,0], data_1003b[30:39,1], data_1003b[30:39,2],c='g')
ax.scatter(data_1003b[40:49,0], data_1003b[40:49,1], data_1003b[40:49,2],c='c')
ax.set_xlabel('Power lambda_1')
ax.set_ylabel('Power lambda_2')
ax.set_zlabel('Power_lambda_3')
plt.title('Features 1003b')
plt.show()

# 1003n
fig=plt.figure(dpi=500)
ax = fig.add_subplot(projection='3d')
ax.scatter(data_1003n[0:9,0], data_1003n[0:9,1], data_1003n[0:9,2],c='k')
ax.scatter(data_1003n[10:19,0], data_1003n[10:19,1], data_1003n[10:19,2],c='r')
ax.scatter(data_1003n[20:29,0], data_1003n[20:29,1], data_1003n[20:29,2],c='b')
ax.scatter(data_1003n[30:39,0], data_1003n[30:39,1], data_1003n[30:39,2],c='g')
ax.scatter(data_1003n[40:49,0], data_1003n[40:49,1], data_1003n[40:49,2],c='c')
ax.set_xlabel('Power lambda_1')
ax.set_ylabel('Power lambda_2')
ax.set_zlabel('Power_lambda_3')
plt.title('Features 1003n')
plt.show()


# 0103b
fig=plt.figure(dpi=500)
ax = fig.add_subplot(projection='3d')
ax.scatter(data_0103b[0:9,0], data_0103b[0:9,1], data_0103b[0:9,2],c='k')
ax.scatter(data_0103b[10:19,0], data_0103b[10:19,1], data_0103b[10:19,2],c='r')
ax.scatter(data_0103b[20:29,0], data_0103b[20:29,1], data_0103b[20:29,2],c='b')
ax.scatter(data_0103b[30:39,0], data_0103b[30:39,1], data_0103b[30:39,2],c='g')
ax.scatter(data_0103b[40:49,0], data_0103b[40:49,1], data_0103b[40:49,2],c='c')
ax.set_xlabel('Phase lambda_1')
ax.set_ylabel('Phase lambda_2')
ax.set_zlabel('Phase_lambda_3')
plt.title('Features 0103b')
plt.show()

# 0103n
fig=plt.figure(dpi=500)
ax = fig.add_subplot(projection='3d')
ax.scatter(data_0103n[0:9,0], data_0103n[0:9,1], data_0103n[0:9,2],c='k')
ax.scatter(data_0103n[10:19,0], data_0103n[10:19,1], data_0103n[10:19,2],c='r')
ax.scatter(data_0103n[20:29,0], data_0103n[20:29,1], data_0103n[20:29,2],c='b')
ax.scatter(data_0103n[30:39,0], data_0103n[30:39,1], data_0103n[30:39,2],c='g')
ax.scatter(data_0103n[40:49,0], data_0103n[40:49,1], data_0103n[40:49,2],c='c')
ax.set_xlabel('Phase lambda_1')
ax.set_ylabel('Phase lambda_2')
ax.set_zlabel('Phase_lambda_3')
plt.title('Features 0103n')
plt.show()

# 0013b
fig=plt.figure(dpi=500)
ax = fig.add_subplot(projection='3d')
ax.scatter(data_0013b[0:9,0], data_0013b[0:9,1], data_0013b[0:9,2],c='k')
ax.scatter(data_0013b[10:19,0], data_0013b[10:19,1], data_0013b[10:19,2],c='r')
ax.scatter(data_0013b[20:29,0], data_0013b[20:29,1], data_0013b[20:29,2],c='b')
ax.scatter(data_0013b[30:39,0], data_0013b[30:39,1], data_0013b[30:39,2],c='g')
ax.scatter(data_0013b[40:49,0], data_0013b[40:49,1], data_0013b[40:49,2],c='c')
ax.set_xlabel('Polarization lambda_1')
ax.set_ylabel('Polarization lambda_2')
ax.set_zlabel('Polarization_lambda_3')
plt.title('Features 0013b')
plt.show()

# 0013n
fig=plt.figure(dpi=500)
ax = fig.add_subplot(projection='3d')
ax.scatter(data_0013n[0:9,0], data_0013n[0:9,1], data_0013n[0:9,2],c='k')
ax.scatter(data_0013n[10:19,0], data_0013n[10:19,1], data_0013n[10:19,2],c='r')
ax.scatter(data_0013n[20:29,0], data_0013n[20:29,1], data_0013n[20:29,2],c='b')
ax.scatter(data_0013n[30:39,0], data_0013n[30:39,1], data_0013n[30:39,2],c='g')
ax.scatter(data_0013n[40:49,0], data_0013n[40:49,1], data_0013n[40:49,2],c='c')
ax.set_xlabel('Polarization lambda_1')
ax.set_ylabel('Polarization lambda_2')
ax.set_zlabel('Polarization_lambda_3')
plt.title('Features 0013n')
plt.show()





# ii) Illustrate features 2D Plots


# 110b
fig=plt.figure(dpi=500)
ax = fig.add_subplot()
ax.scatter(data_110b[0:9,0], data_110b[0:9,1], c='k')
ax.scatter(data_110b[10:19,0], data_110b[10:19,1], c='r')
ax.scatter(data_110b[20:29,0], data_110b[20:29,1], c='b')
ax.scatter(data_110b[30:39,0], data_110b[30:39,1], c='g')
ax.scatter(data_110b[40:49,0], data_110b[40:49,1], c='c')
ax.set_xlabel('Power')
ax.set_ylabel('Phase')
plt.title('Features 110b')
plt.show()


# 110n
fig=plt.figure(dpi=500)
ax = fig.add_subplot()
ax.scatter(data_110n[0:9,0], data_110n[0:9,1], c='k')
ax.scatter(data_110n[10:19,0], data_110n[10:19,1], c='r')
ax.scatter(data_110n[20:29,0], data_110n[20:29,1], c='b')
ax.scatter(data_110n[30:39,0], data_110n[30:39,1], c='g')
ax.scatter(data_110n[40:49,0], data_110n[40:49,1], c='c')
ax.set_xlabel('Power')
ax.set_ylabel('Phase')
plt.title('Features 110n')
plt.show()

# 101b
fig=plt.figure(dpi=500)
ax = fig.add_subplot()
ax.scatter(data_101b[0:9,0], data_110b[0:9,1], c='k')
ax.scatter(data_101b[10:19,0], data_101b[10:19,1], c='r')
ax.scatter(data_101b[20:29,0], data_101b[20:29,1], c='b')
ax.scatter(data_101b[30:39,0], data_101b[30:39,1], c='g')
ax.scatter(data_101b[40:49,0], data_101b[40:49,1], c='c')
ax.set_xlabel('Power')
ax.set_ylabel('Polarization')
plt.title('Features 101b')
plt.show()


# 101n
fig=plt.figure(dpi=500)
ax = fig.add_subplot()
ax.scatter(data_101n[0:9,0], data_101n[0:9,1], c='k')
ax.scatter(data_101n[10:19,0], data_101n[10:19,1], c='r')
ax.scatter(data_101n[20:29,0], data_101n[20:29,1], c='b')
ax.scatter(data_101n[30:39,0], data_101n[30:39,1], c='g')
ax.scatter(data_101n[40:49,0], data_101n[40:49,1], c='c')
ax.set_xlabel('Power')
ax.set_ylabel('Polarization')
plt.title('Features 101n')
plt.show()

# 011b
fig=plt.figure(dpi=500)
ax = fig.add_subplot()
ax.scatter(data_011b[0:9,0], data_011b[0:9,1], c='k')
ax.scatter(data_011b[10:19,0], data_011b[10:19,1], c='r')
ax.scatter(data_011b[20:29,0], data_011b[20:29,1], c='b')
ax.scatter(data_011b[30:39,0], data_011b[30:39,1], c='g')
ax.scatter(data_011b[40:49,0], data_011b[40:49,1], c='c')
ax.set_xlabel('Phase')
ax.set_ylabel('Polarization')
plt.title('Features 011b')
plt.show()


# 011n
fig=plt.figure(dpi=500)
ax = fig.add_subplot()
ax.scatter(data_011n[0:9,0], data_011n[0:9,1], c='k')
ax.scatter(data_011n[10:19,0], data_011n[10:19,1], c='r')
ax.scatter(data_011n[20:29,0], data_011n[20:29,1], c='b')
ax.scatter(data_011n[30:39,0], data_011n[30:39,1], c='g')
ax.scatter(data_011n[40:49,0], data_011n[40:49,1], c='c')
ax.set_xlabel('Phase')
ax.set_ylabel('Polarization')
plt.title('Features 011n')
plt.show()









# 100b
fig=plt.figure(dpi=500)
ax = fig.add_subplot()
ax.scatter(data_100b[0:9,0], data_100b[0:9,1], c='k')
ax.scatter(data_100b[10:19,0], data_100b[10:19,1], c='r')
ax.scatter(data_100b[20:29,0], data_100b[20:29,1], c='b')
ax.scatter(data_100b[30:39,0], data_100b[30:39,1], c='g')
ax.scatter(data_100b[40:49,0], data_100b[40:49,1], c='c')
ax.set_xlabel('Power lambda_1')
ax.set_ylabel('Power lambda_2')
plt.title('Features 100b')
plt.show()


# 100n
fig=plt.figure(dpi=500)
ax = fig.add_subplot()
ax.scatter(data_100n[0:9,0], data_100n[0:9,1], c='k')
ax.scatter(data_100n[10:19,0], data_100n[10:19,1], c='r')
ax.scatter(data_100n[20:29,0], data_100n[20:29,1], c='b')
ax.scatter(data_100n[30:39,0], data_100n[30:39,1], c='g')
ax.scatter(data_100n[40:49,0], data_100n[40:49,1], c='c')
ax.set_xlabel('Power lambda_1')
ax.set_ylabel('Power lambda_2')
plt.title('Features 100n')
plt.show()


# 010b
fig=plt.figure(dpi=500)
ax = fig.add_subplot()
ax.scatter(data_010b[0:9,0], data_010b[0:9,1], c='k')
ax.scatter(data_010b[10:19,0], data_010b[10:19,1], c='r')
ax.scatter(data_010b[20:29,0], data_010b[20:29,1], c='b')
ax.scatter(data_010b[30:39,0], data_010b[30:39,1], c='g')
ax.scatter(data_010b[40:49,0], data_010b[40:49,1], c='c')
ax.set_xlabel('Phase lambda_1')
ax.set_ylabel('Phase lambda_2')
plt.title('Features 010b')
plt.show()


# 010n
fig=plt.figure(dpi=500)
ax = fig.add_subplot()
ax.scatter(data_010n[0:9,0], data_010n[0:9,1], c='k')
ax.scatter(data_010n[10:19,0], data_010n[10:19,1], c='r')
ax.scatter(data_010n[20:29,0], data_010n[20:29,1], c='b')
ax.scatter(data_010n[30:39,0], data_010n[30:39,1], c='g')
ax.scatter(data_010n[40:49,0], data_010n[40:49,1], c='c')
ax.set_xlabel('Phase lambda_1')
ax.set_ylabel('Phase lambda_2')
plt.title('Features 010n')
plt.show()

# 001b
fig=plt.figure(dpi=500)
ax = fig.add_subplot()
ax.scatter(data_001b[0:9,0], data_001b[0:9,1], c='k')
ax.scatter(data_001b[10:19,0], data_001b[10:19,1], c='r')
ax.scatter(data_001b[20:29,0], data_001b[20:29,1], c='b')
ax.scatter(data_001b[30:39,0], data_001b[30:39,1], c='g')
ax.scatter(data_001b[40:49,0], data_001b[40:49,1], c='c')
ax.set_xlabel('Polarization lambda_1')
ax.set_ylabel('Polarization lambda_2')
plt.title('Features 010b')
plt.show()


# 001n
fig=plt.figure(dpi=500)
ax = fig.add_subplot()
ax.scatter(data_001n[0:9,0], data_001n[0:9,1], c='k')
ax.scatter(data_001n[10:19,0], data_001n[10:19,1], c='r')
ax.scatter(data_001n[20:29,0], data_001n[20:29,1], c='b')
ax.scatter(data_001n[30:39,0], data_001n[30:39,1], c='g')
ax.scatter(data_001n[40:49,0], data_001n[40:49,1], c='c')
ax.set_xlabel('Polarization lambda_1')
ax.set_ylabel('Polarization lambda_2')
plt.title('Features 010n')
plt.show()





