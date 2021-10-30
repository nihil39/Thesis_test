import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.ticker import AutoMinorLocator
#from matplotlib import gridspec
#import scipy as sp
import random
import scipy as sp
from scipy import stats

#array_msd_T044 = np.load('msd_completo_T044_P293.npy')
#array_msd_T047 = np.load('msd_completo_T047_P224.npy')
#array_msd_T050 = np.load('msd_completo_T050_P155.npy')
#array_msd_T056 = np.load('msd_completo_T056_P017.npy') #[configurazioni, (questo campo deve scorrere coi :),  0 = tempi / 1 = msdA / 2 = msdB]

array_msd_T044_medie = np.load('msd_T044_P293_medie.npy')
array_msd_T047_medie = np.load('msd_T047_P224_medie.npy')
array_msd_T050_medie = np.load('msd_T050_P155_medie.npy')
array_msd_T056_medie = np.load('msd_T056_P017_medie.npy') #[configurazioni, (questo campo deve scorrere coi :),  0 = tempi / 1 = msdA / 2 = msdB]

array_msd_T052_NT_medie = np.load('msd_T052_P109_NT_medie.npy') #[configurazioni, (questo campo deve scorrere coi :),  0 = tempi / 1 = msdA / 2 = msdB]

#array_msd_T052_medie = np.load('msd_T052_P109_NT_medie.npy') #[configurazioni, (questo campo deve scorrere coi :),  0 = tempi / 1 = msdA / 2 = msdB]

#predizioni_msd_T044 = np.load('predictions_k20_e30_bs16_cpMax_T044_P293.npy')
#predizioni_msd_T047 = np.load('predictions_k20_e30_bs16_cpMax_T047_P224.npy')
#predizioni_msd_T050 = np.load('predictions_k20_e30_bs16_cpMax_T050_P155.npy')
#predizioni_msd_T056 = np.load('predictions_k20_e30_bs16_cpMax_T056_P017.npy')

predizioni_msd_T052_NT_all_conf = np.load('../DGCNN/Tutte_T052/predictions_k20_e30_bs16_cpMax_T052.npy')

fig = plt.figure()
fig.suptitle('MSD mean and predictions on the new conf T = 0.52, P = 1.09\nDGCNN trained on the other four configurations', fontsize = 14, c = 'darkgrey')


#ax1 = fig.add_subplot(1, 2, 1)
#ax2 = fig.add_subplot(1, 2, 2)

ax2 = fig.add_subplot(1, 1, 1)

configurazione = random.randint(0, 7999)
msdA = 1
msdB = 2

# l'asse x, il primo campo, i tempi, sono sempre gli stessi per tutte le configurazioni, va bene quindi array_msd_T044[0, :, 0]  
# Why don't you V-Block select 4 columns and hit c, change the values and then hit esc?

#ax1.plot(array_msd_T044[0, :, 0], array_msd_T044[configurazione, :, msdA], color = 'tab:blue', label = "T = 0.44, P = 2.93" ) #ax1.scatter per lo scatter plot
#ax1.plot(array_msd_T047[0, :, 0], array_msd_T047[configurazione, :, msdA], color = 'mediumseagreen', label = "T = 0.47, P = 2.24" )
#ax1.plot(array_msd_T047[0, :, 0], array_msd_T050[configurazione, :, msdA], color = 'orange', label = "T = 0.50, P = 1.55")
#ax1.plot(array_msd_T047[0, :, 0], array_msd_T056[configurazione, :, msdA], color = 'tab:red', label = "T = 0.56, P = 0.17")

ax2.plot(array_msd_T044_medie[:, 0], array_msd_T044_medie[:, msdA], color = 'tab:blue', label = "T = 0.44, P = 2.93" ) #ax1.scatter per lo scatter plot
ax2.plot(array_msd_T047_medie[:, 0], array_msd_T047_medie[:, msdA], color = 'mediumseagreen', label = "T = 0.47, P = 2.24" )
ax2.plot(array_msd_T050_medie[:, 0], array_msd_T050_medie[:, msdA], color = 'orange', label = "T = 0.50, P = 1.55")
ax2.plot(array_msd_T056_medie[:, 0], array_msd_T056_medie[:, msdA], color = 'tab:red', label = "T = 0.56, P = 0.17")

ax2.plot(array_msd_T052_NT_medie[:, 0], array_msd_T052_NT_medie[:, msdA], color = 'lightcoral', label = "T = 0.52, P = 1.09")

#ax2.plot(array_msd_T052_medie[:, 0], array_msd_T052_medie[:, msdA], color = 'black', label = "T = 0.52, P = 1.09")

# 0.005, 1.63, 500 sono i valori temporali a cui e' stata fatta la predizione dalla NN, corrispondono a 0, 100 e 199


## T = 0.52, P = 1.09
ax2.plot(1.63 , predizioni_msd_T052_NT_all_conf[:,0].mean(), color = 'lightcoral', marker = '8')
ax2.plot(500  , predizioni_msd_T052_NT_all_conf[:,1].mean(), color = 'lightcoral', marker = '8')


#ax1.set_title(f'One configuration')
#ax1.legend()
#ax1.set_xlabel('time (#iterations)')
#ax1.set_ylabel(r'$\langle r^2(t) \rangle$', fontsize = 14) #r' ' per scrivere in latex tra gli apici
#ax1.set_xscale('log')
#ax1.set_yscale('log')


#ax2.set_title(f'Mean')
ax2.legend()
ax2.set_xlabel('time (#iterations)')
ax2.set_ylabel(r'$\langle r^2(t) \rangle$', fontsize = 14) #r' ' per scrivere in latex tra gli apici
ax2.set_xscale('log')
ax2.set_yscale('log')

ax2.grid(linestyle = '--', linewidth = 0.5)
plt.show()

