import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.ticker import AutoMinorLocator
#from matplotlib import gridspec
import scipy as sp

array_msd_T044 = np.load('msd_completo_T044_P293.npy')
array_msd_T047 = np.load('msd_completo_T047_P224.npy')
array_msd_T050 = np.load('msd_completo_T050_P155.npy')
array_msd_T056 = np.load('msd_completo_T056_P017.npy') #[configurazioni, (questo campo deve scorrere coi :), tempi / msdA / msdB]

fig = plt.figure()
fig.suptitle('MSD', fontsize = 14, c='orange')

ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)


ax1.set_title(f'T = 0.44 P = 2.93')
ax2.set_title(f'T = 0.47 P = 2.24')
ax3.set_title(f'T = 0.50 P = 1.55')
ax4.set_title(f'T = 0.56 P = 0.17')


ax1.plot(array_msd_T044[0, :, 0], array_msd_T044[0, :, 1], color = 'crimson') #ax1.scatter per lo scatter plot
ax1.set_xscale('log')
ax1.set_yscale('log')

ax2.plot(array_msd_T047[0, :, 0], array_msd_T047[0, :, 1], color = 'green')
ax2.set_xscale('log')
ax2.set_yscale('log')

ax3.plot(array_msd_T050[0, :, 0], array_msd_T050[0, :, 1], color = 'blue')
ax3.set_xscale('log')
ax3.set_yscale('log')

ax4.plot(array_msd_T056[0, :, 0], array_msd_T056[0, :, 1], color = 'magenta')
ax4.set_xscale('log')
ax4.set_yscale('log')

plt.show()
