"""
The goal of this script is to illustrate the msmc data.
For this, do the following:
    1. Imports and definitions
    2. Plot Phase 10p data
    3. Plot Power 10p data
    4. Plot Polarization 10p data
"""


"""
    1. Imports and definitions -----------------------------------------------
"""


# i) Imports

import numpy as np
import matplotlib.pyplot as plt


# ii) Load data 

Phase_10p=np.load('Material_classification_python/Phase_10p.npy')
Power_10p=np.load('Material_classification_python/Power_10p.npy')
Polarization_10p=np.load('Material_classification_python/Polarization_10p.npy')
        
        
        
"""
    2. Plot Phase 10p data ----------------------------------------------------
"""


# i) Define figure properties

fig = plt.figure(dpi=200,constrained_layout=True)
gs = fig.add_gridspec(3, 5)


# ii) Construct subplots

f_ax00 = fig.add_subplot(gs[0,0])
plt.plot(Phase_10p[0].T)
plt.axis('off')
plt.title('Phase 10p')

f_ax01 = fig.add_subplot(gs[0,1])
plt.plot(Phase_10p[1].T)
plt.axis('off')

f_ax02 = fig.add_subplot(gs[0,2])
plt.plot(Phase_10p[2].T)
plt.axis('off')

f_ax03 = fig.add_subplot(gs[0,3])
plt.plot(Phase_10p[3].T)
plt.axis('off')

f_ax04 = fig.add_subplot(gs[0,4])
plt.plot(Phase_10p[4].T)
plt.axis('off')




"""
    3. Plot Power 10p data ---------------------------------------------------
"""


# i) Construct subplots

f_ax10 = fig.add_subplot(gs[1,0])
plt.plot(Power_10p[0].T)
plt.axis('off')
plt.title('Power 10p')

f_ax11 = fig.add_subplot(gs[1,1])
plt.plot(Power_10p[1].T)
plt.axis('off')

f_ax12 = fig.add_subplot(gs[1,2])
plt.plot(Power_10p[2].T)
plt.axis('off')

f_ax13 = fig.add_subplot(gs[1,3])
plt.plot(Power_10p[3].T)
plt.axis('off')

f_ax14 = fig.add_subplot(gs[1,4])
plt.plot(Power_10p[4].T)
plt.axis('off')



"""
    4. Plot Polarization 10p data ----------------------------------------------
"""


# i) Construct subplots

f_ax20 = fig.add_subplot(gs[2,0])
plt.plot(Polarization_10p[0].T)
plt.axis('off')
plt.title('Polarization 10p')

f_ax21 = fig.add_subplot(gs[2,1])
plt.plot(Polarization_10p[1].T)
plt.axis('off')

f_ax22 = fig.add_subplot(gs[2,2])
plt.plot(Polarization_10p[2].T)
plt.axis('off')

f_ax23 = fig.add_subplot(gs[2,3])
plt.plot(Polarization_10p[3].T)
plt.axis('off')

f_ax24 = fig.add_subplot(gs[2,4])
plt.plot(Polarization_10p[4].T)
plt.axis('off')






















































