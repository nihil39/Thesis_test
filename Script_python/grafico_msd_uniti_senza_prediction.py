import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.ticker import AutoMinorLocator
#from matplotlib import gridspec
import random
import scipy as sp
from scipy import stats

#array_msd_T044 = np.load('msd_completo_T044_P293.npy')
#array_msd_T047 = np.load('msd_completo_T047_P224.npy')
#array_msd_T050 = np.load('msd_completo_T050_P155.npy')
#array_msd_T056 = np.load('msd_completo_T056_P017.npy') #[configurazioni, (questo campo deve scorrere coi :),  0 = tempi / 1 = particle_type / 2 = msdB]

array_msd_T044_medie = np.load('msd_T044_P293_medie.npy')
array_msd_T045_NT_medie = np.load('msd_T045_P270_NT_medie.npy')
array_msd_T047_medie = np.load('msd_T047_P224_medie.npy')
array_msd_T049_NT_medie = np.load('msd_T049_P178_NT_medie.npy')
array_msd_T050_medie = np.load('msd_T050_P155_medie.npy')
array_msd_T052_NT_medie = np.load('msd_T052_P109_NT_medie.npy') #[configurazioni, (questo campo deve scorrere coi :),  0 = tempi / 1 = particle_type / 2 = msdB]
array_msd_T056_medie = np.load('msd_T056_P017_medie.npy') #[configurazioni, (questo campo deve scorrere coi :),  0 = tempi / 1 = msdA / 2 = msdB]

particle_type = 1 # 1 = A, 2 = B

fig = plt.figure()
fig.suptitle('MSD mean on all the configurations', fontsize = 14, c = 'darkgrey')

#ax1 = fig.add_subplot(1, 2, 1)
#ax2 = fig.add_subplot(1, 2, 2)

ax2 = fig.add_subplot(1, 1, 1)

configurazione = random.randint(0, 7999)

# l'asse x, il primo campo, i tempi, sono sempre gli stessi per tutte le configurazioni, va bene quindi array_msd_T044[0, :, 0]  
# Why don't you V-Block select 4 columns and hit c, change the values and then hit esc?

#ax1.plot(array_msd_T044[0, :, 0], array_msd_T044[configurazione, :, particle_type], color = 'tab:blue', label = "T = 0.44, P = 2.93" ) #ax1.scatter per lo scatter plot
#ax1.plot(array_msd_T047[0, :, 0], array_msd_T047[configurazione, :, particle_type], color = 'mediumseagreen', label = "T = 0.47, P = 2.24" )
#ax1.plot(array_msd_T047[0, :, 0], array_msd_T050[configurazione, :, particle_type], color = 'orange', label = "T = 0.50, P = 1.55")
#ax1.plot(array_msd_T047[0, :, 0], array_msd_T056[configurazione, :, particle_type], color = 'tab:red', label = "T = 0.56, P = 0.17")

ax2.plot(array_msd_T056_medie[:, 0], array_msd_T056_medie[:, particle_type], color = 'tab:red', label = "T = 0.56, P = 0.17")
ax2.plot(array_msd_T052_NT_medie[:, 0], array_msd_T052_NT_medie[:, particle_type], color = 'coral', linestyle = 'dashed', label = "T = 0.52, P = 1.09 NT")
ax2.plot(array_msd_T050_medie[:, 0], array_msd_T050_medie[:, particle_type], color = 'orange', label = "T = 0.50, P = 1.55")
ax2.plot(array_msd_T049_NT_medie[:, 0], array_msd_T049_NT_medie[:, particle_type], color = 'gold', linestyle = 'dashed', label = "T = 0.49, P = 1.78 NT" )
ax2.plot(array_msd_T047_medie[:, 0], array_msd_T047_medie[:, particle_type], color = 'mediumseagreen', label = "T = 0.47, P = 2.24" )
ax2.plot(array_msd_T045_NT_medie[:, 0], array_msd_T045_NT_medie[:, particle_type], color = 'cornflowerblue', linestyle = 'dashed', label = "T = 0.45, P = 2.70 NT" ) #ax1.scatter per lo scatter plot
ax2.plot(array_msd_T044_medie[:, 0], array_msd_T044_medie[:, particle_type], color = 'tab:blue', label = "T = 0.44, P = 2.93" ) #ax1.scatter per lo scatter plot


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

