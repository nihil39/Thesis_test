import numpy as np
import scipy as sp
from scipy import stats

array_msd_T044 = np.load('msd_completo_T044_P293.npy')
array_mu_sigma_T044 = np.zeros((200,4))

#In the array_mu_sigma are stored the mean and variance (gaussian fit) of the msd distribution for every time. [tempi, (0: media partA, 1: sigma partA, 2: media partB, 3: sigma partB)]

for j in range(200):
   array_mu_sigma_T044[j,0], array_mu_sigma_T044[j,1] = sp.stats.norm.fit(array_msd_T044[:, j, 1])
   array_mu_sigma_T044[j,2], array_mu_sigma_T044[j,3] = sp.stats.norm.fit(array_msd_T044[:, j, 2])
    
np.save('array_mu_sigma_T044.npy', array_mu_sigma_T044)


