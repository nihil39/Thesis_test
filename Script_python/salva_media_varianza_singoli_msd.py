import numpy as np
import scipy as sp
from scipy import stats

array_msd_T044 = np.load('msd_completo_T044_P293.npy')
array_mu_sigma_T044 = np.zeros((200,5))

array_msd_T045_NT = np.load('msd_completo_T045_P270_NT.npy')
array_mu_sigma_T045_NT = np.zeros((200,5))

array_msd_T047 = np.load('msd_completo_T047_P224.npy')
array_mu_sigma_T047 = np.zeros((200,5))

array_msd_T049_NT = np.load('msd_completo_T049_P178_NT.npy')
array_mu_sigma_T049_NT = np.zeros((200,5))

array_msd_T050 = np.load('msd_completo_T050_P155.npy')
array_mu_sigma_T050 = np.zeros((200,5))

array_msd_T052_NT = np.load('msd_completo_T052_P109_NT.npy')
array_mu_sigma_T052_NT = np.zeros((200,5))

array_msd_T056 = np.load('msd_completo_T056_P017.npy')
array_mu_sigma_T056 = np.zeros((200,5))

#In the array_mu_sigma are stored the mean and variance (gaussian fit) of the msd distribution for every time. [tempi, (0: timestep, 1: media partA, 2: sigma partA, 3: media partB)4: sigma partB]

for j in range(200):
    array_mu_sigma_T044[j, 0] = array_msd_T044[0, j, 0] #copia i tempi 
    array_mu_sigma_T044[j, 1], array_mu_sigma_T044[j, 2] = sp.stats.norm.fit(array_msd_T044[:, j, 1])
    array_mu_sigma_T044[j, 3], array_mu_sigma_T044[j, 4] = sp.stats.norm.fit(array_msd_T044[:, j, 2])
    
    array_mu_sigma_T045_NT[j, 0] = array_msd_T045_NT[0, j, 0] #copia i tempi 
    array_mu_sigma_T045_NT[j, 1], array_mu_sigma_T045_NT[j, 2] = sp.stats.norm.fit(array_msd_T045_NT[:, j, 1])
    array_mu_sigma_T045_NT[j, 3], array_mu_sigma_T045_NT[j, 4] = sp.stats.norm.fit(array_msd_T045_NT[:, j, 2])
    
    array_mu_sigma_T047[j, 0] = array_msd_T047[0, j, 0] #copia i tempi 
    array_mu_sigma_T047[j, 1], array_mu_sigma_T047[j, 2] = sp.stats.norm.fit(array_msd_T047[:, j, 1])
    array_mu_sigma_T047[j, 3], array_mu_sigma_T047[j, 4] = sp.stats.norm.fit(array_msd_T047[:, j, 2])
    
    array_mu_sigma_T049_NT[j, 0] = array_msd_T049_NT[0, j, 0] #copia i tempi 
    array_mu_sigma_T049_NT[j, 1], array_mu_sigma_T049_NT[j, 2] = sp.stats.norm.fit(array_msd_T049_NT[:, j, 1])
    array_mu_sigma_T049_NT[j, 3], array_mu_sigma_T049_NT[j, 4] = sp.stats.norm.fit(array_msd_T049_NT[:, j, 2])
    
    array_mu_sigma_T050[j, 0] = array_msd_T050[0, j, 0] #copia i tempi 
    array_mu_sigma_T050[j, 1], array_mu_sigma_T050[j, 2] = sp.stats.norm.fit(array_msd_T050[:, j, 1])
    array_mu_sigma_T050[j, 3], array_mu_sigma_T050[j, 4] = sp.stats.norm.fit(array_msd_T050[:, j, 2])
    
    array_mu_sigma_T052_NT[j, 0] = array_msd_T052_NT[0, j, 0] #copia i tempi 
    array_mu_sigma_T052_NT[j, 1], array_mu_sigma_T052_NT[j, 2] = sp.stats.norm.fit(array_msd_T052_NT[:, j, 1])
    array_mu_sigma_T052_NT[j, 3], array_mu_sigma_T052_NT[j, 4] = sp.stats.norm.fit(array_msd_T052_NT[:, j, 2])
    
    array_mu_sigma_T056[j, 0] = array_msd_T056[0, j, 0] #copia i tempi 
    array_mu_sigma_T056[j, 1], array_mu_sigma_T056[j, 2] = sp.stats.norm.fit(array_msd_T056[:, j, 1])
    array_mu_sigma_T056[j, 3], array_mu_sigma_T056[j, 4] = sp.stats.norm.fit(array_msd_T056[:, j, 2])

np.save('array_mu_sigma_T044.npy', array_mu_sigma_T044)
np.save('array_mu_sigma_T045_NT.npy', array_mu_sigma_T045_NT)
np.save('array_mu_sigma_T047.npy', array_mu_sigma_T047)
np.save('array_mu_sigma_T049_NT.npy', array_mu_sigma_T049_NT)
np.save('array_mu_sigma_T050.npy', array_mu_sigma_T050)
np.save('array_mu_sigma_T052_NT.npy', array_mu_sigma_T052_NT)
np.save('array_mu_sigma_T056.npy', array_mu_sigma_T056)

