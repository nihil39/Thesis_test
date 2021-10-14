import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.ticker import AutoMinorLocator
#from matplotlib import gridspec
#import scipy as sp

array_msd_T044 = np.load('msd_completo_T044_P293.npy')
array_msd_T047 = np.load('msd_completo_T047_P224.npy')
array_msd_T050 = np.load('msd_completo_T050_P155.npy')
array_msd_T056 = np.load('msd_completo_T056_P017.npy') #[configurazioni, (questo campo deve scorrere coi :),  0 = tempi / 1 = msdA / 2 = msdB]

fig = plt.figure()
fig.suptitle('MSD', fontsize = 14, c='darkgrey')

ax1 = fig.add_subplot(1, 1, 1)

configurazione = 7
msdA = 1
msdB = 2

# l'asse x, il primo campo, i tempi, sono sempre gli stessi per tutte le configurazioni, va bene quindi array_msd_T044[0, :, 0]  
# Why don't you V-Block select 4 columns and hit c, change the values and then hit esc?

ax1.plot(array_msd_T044[0, :, 0], array_msd_T044[configurazione, :, msdA], color = 'tab:blue', label = "T = 0.44, P = 2.93" ) #ax1.scatter per lo scatter plot
ax1.plot(array_msd_T047[0, :, 0], array_msd_T047[configurazione, :, msdA], color = 'mediumseagreen', label = "T = 0.47, P = 2.24" )
ax1.plot(array_msd_T047[0, :, 0], array_msd_T050[configurazione, :, msdA], color = 'orange', label = "T = 0.50, P = 1.55")
ax1.plot(array_msd_T047[0, :, 0], array_msd_T056[configurazione, :, msdA], color = 'tab:red', label = "T = 0.56, P = 0.17")

ax1.legend()
ax1.set_xlabel('time (#iterations)')
ax1.set_ylabel(r'$\langle r^2(t) \rangle$', fontsize = 14) #r' ' per scrivere in latex tra gli apici

ax1.set_xscale('log')
ax1.set_yscale('log')

plt.show()

