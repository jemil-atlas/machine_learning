"""
The goal of this script is to import the ms_mc data into python vectors and store
them for later processing.
"""

"""
    1. Load data -------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt

# msmc_kl= multispectral material classification data k,l
#           k= material number, l=1,2,3 phase or power or polarization
# material 1: cardboard
# material 2: foam
# material 3: plaster
# material 4: plastic
# material 5: woodboard


# Phase

msmcdata_11=np.genfromtxt('/home/jemil/Desktop/Programming/Python/Optimization/Material_Signature_Classification/Material_classification/Phase_spectral_signatures/cardboard_10p_Distance_mm.txt',delimiter=',').T
msmcdata_21=np.genfromtxt('/home/jemil/Desktop/Programming/Python/Optimization/Material_Signature_Classification/Material_classification/Phase_spectral_signatures/foam_10p_Distance_mm.txt',delimiter=',').T
msmcdata_31=np.genfromtxt('/home/jemil/Desktop/Programming/Python/Optimization/Material_Signature_Classification/Material_classification/Phase_spectral_signatures/plaster_10p_Distance_mm.txt',delimiter=',').T
msmcdata_41=np.genfromtxt('/home/jemil/Desktop/Programming/Python/Optimization/Material_Signature_Classification/Material_classification/Phase_spectral_signatures/plastic_10p_Distance_mm.txt',delimiter=',').T
msmcdata_51=np.genfromtxt('/home/jemil/Desktop/Programming/Python/Optimization/Material_Signature_Classification/Material_classification/Phase_spectral_signatures/woodboard_10p_Distance_mm.txt',delimiter=',').T


# Power

msmcdata_12=np.genfromtxt('/home/jemil/Desktop/Programming/Python/Optimization/Material_Signature_Classification/Material_classification/Power_spectral_signatures/cardboard_10p_Reflectance_cali.txt',delimiter=',').T
msmcdata_22=np.genfromtxt('/home/jemil/Desktop/Programming/Python/Optimization/Material_Signature_Classification/Material_classification/Power_spectral_signatures/foam_10p_Reflectance_cali.txt',delimiter=',').T
msmcdata_32=np.genfromtxt('/home/jemil/Desktop/Programming/Python/Optimization/Material_Signature_Classification/Material_classification/Power_spectral_signatures/plaster_10p_Reflectance_cali.txt',delimiter=',').T
msmcdata_42=np.genfromtxt('/home/jemil/Desktop/Programming/Python/Optimization/Material_Signature_Classification/Material_classification/Power_spectral_signatures/plastic_10p_Reflectance_cali.txt',delimiter=',').T
msmcdata_52=np.genfromtxt('/home/jemil/Desktop/Programming/Python/Optimization/Material_Signature_Classification/Material_classification/Power_spectral_signatures/woodboard_10p_Reflectance_cali.txt',delimiter=',').T


# Polarization

msmcdata_13=np.genfromtxt('/home/jemil/Desktop/Programming/Python/Optimization/Material_Signature_Classification/Material_classification/Polarization_spectral_signatures/cardboard_10p_DoLP_LaSour.txt',delimiter=',').T
msmcdata_23=np.genfromtxt('/home/jemil/Desktop/Programming/Python/Optimization/Material_Signature_Classification/Material_classification/Polarization_spectral_signatures/foam_10p_DoLP_LaSour.txt',delimiter=',').T
msmcdata_33=np.genfromtxt('/home/jemil/Desktop/Programming/Python/Optimization/Material_Signature_Classification/Material_classification/Polarization_spectral_signatures/plaster_10p_DoLP_LaSour.txt',delimiter=',').T
msmcdata_43=np.genfromtxt('/home/jemil/Desktop/Programming/Python/Optimization/Material_Signature_Classification/Material_classification/Polarization_spectral_signatures/plastic_10p_DoLP_LaSour.txt',delimiter=',').T
msmcdata_53=np.genfromtxt('/home/jemil/Desktop/Programming/Python/Optimization/Material_Signature_Classification/Material_classification/Polarization_spectral_signatures/woodboard_10p_DoLP_LaSour.txt',delimiter=',').T


# Aux
wavelengths_7=np.genfromtxt('/home/jemil/Desktop/Programming/Python/Optimization/Material_Signature_Classification/Old_processing/Material_Signatures_Pack1/WL_40nm.txt', delimiter=',')
wavelengths_33=np.genfromtxt('/home/jemil/Desktop/Programming/Python/Optimization/Material_Signature_Classification/Old_processing/Material_Signatures_Pack1/WL_10nm.txt', delimiter=',')

wavelength_vector=np.hstack((wavelengths_7,wavelengths_33))



"""
    2. Assemble and save -----------------------------------------------------
"""


# Assemble
Phase_10p=np.array([msmcdata_11, msmcdata_21, msmcdata_31, msmcdata_41, msmcdata_51])
Power_10p=np.array([msmcdata_12, msmcdata_22, msmcdata_32, msmcdata_42, msmcdata_52])
Polarization_10p=np.array([msmcdata_13, msmcdata_23, msmcdata_33, msmcdata_43, msmcdata_53])

# Save
np.save('/home/jemil/Desktop/Programming/Python/Optimization/Material_Signature_Classification/Material_classification_python/Phase_10p', Phase_10p)
np.save('/home/jemil/Desktop/Programming/Python/Optimization/Material_Signature_Classification/Material_classification_python/Power_10p', Power_10p)
np.save('/home/jemil/Desktop/Programming/Python/Optimization/Material_Signature_Classification/Material_classification_python/Polarization_10p', Polarization_10p)
np.save('/home/jemil/Desktop/Programming/Python/Optimization/Material_Signature_Classification/Material_classification_python/wavelength_vector', wavelength_vector)




