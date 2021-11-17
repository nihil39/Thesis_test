import numpy as np
import matplotlib.pyplot as plt

array_mu_sigma_T044 = np.load('array_mu_sigma_T044.npy')
array_mu_sigma_T045_NT = np.load('array_mu_sigma_T045_NT.npy')
array_mu_sigma_T047 = np.load('array_mu_sigma_T047.npy')
array_mu_sigma_T049_NT = np.load('array_mu_sigma_T049_NT.npy')
array_mu_sigma_T050 = np.load('array_mu_sigma_T050.npy')
array_mu_sigma_T052_NT = np.load('array_mu_sigma_T052_NT.npy')
array_mu_sigma_T056 = np.load('array_mu_sigma_T056.npy')
#array_mu_sigma syntax [tempi, (0: timestep, 1: media partA, 2: sigma partA, 3: media partB, 4: sigma partB]

error_T044 = array_mu_sigma_T044[:, 2] 
error_T045_NT = array_mu_sigma_T045_NT[:, 2] 
error_T047 = array_mu_sigma_T047[:, 2] 
error_T049_NT = array_mu_sigma_T049_NT[:, 2] 
error_T050 = array_mu_sigma_T050[:, 2] 
error_T052_NT = array_mu_sigma_T052_NT[:, 2] 
error_T056 = array_mu_sigma_T056[:, 2] 

fig = plt.figure()
fig.suptitle('MSD with shaded error region', fontsize = 14, c = 'darkgrey')

ax2 = fig.add_subplot(1, 1, 1)

ax2.plot(array_mu_sigma_T056[:, 0], array_mu_sigma_T056[:, 1], color = 'tab:red', label = "T = 0.56, P = 0.17" ) #ax1.scatter per lo scatter plot
ax2.fill_between(array_mu_sigma_T056[:, 0], array_mu_sigma_T056[:, 1] + error_T056, array_mu_sigma_T056[:, 1] - error_T056, color = 'indianred',linestyle = 'dashdot', edgecolor = 'slategrey') #ax1.scatter per lo scatter plot

ax2.plot(array_mu_sigma_T052_NT[:, 0], array_mu_sigma_T052_NT[:, 1], color = 'coral', label = "T = 0.52, P = 1.09 NT" ) 
ax2.fill_between(array_mu_sigma_T052_NT[:, 0], array_mu_sigma_T052_NT[:, 1] + error_T052_NT, array_mu_sigma_T052_NT[:, 1] - error_T052_NT, color = 'mistyrose',linestyle = 'dashdot', edgecolor = 'slategrey') 

ax2.plot(array_mu_sigma_T050[:, 0], array_mu_sigma_T050[:, 1], color = 'orange', label = "T = 0.50, P = 1.55" ) 
ax2.fill_between(array_mu_sigma_T050[:, 0], array_mu_sigma_T050[:, 1] + error_T050, array_mu_sigma_T050[:, 1] - error_T050, color = 'wheat',linestyle = 'dashdot', edgecolor = 'slategrey') 

ax2.plot(array_mu_sigma_T049_NT[:, 0], array_mu_sigma_T049_NT[:, 1], color = 'gold', label = "T = 0.49, P = 1.78 NT" )
ax2.fill_between(array_mu_sigma_T049_NT[:, 0], array_mu_sigma_T049_NT[:, 1] + error_T049_NT, array_mu_sigma_T049_NT[:, 1] - error_T049_NT, color = 'xkcd:cream',linestyle = 'dashdot', edgecolor = 'slategrey') 

ax2.plot(array_mu_sigma_T047[:, 0], array_mu_sigma_T047[:, 1], color = 'mediumseagreen', label = "T = 0.47, P = 2.24" ) 
ax2.fill_between(array_mu_sigma_T047[:, 0], array_mu_sigma_T047[:, 1] + error_T047, array_mu_sigma_T047[:, 1] - error_T047, color = 'mediumspringgreen',linestyle = 'dashdot',  edgecolor = 'slategrey') 

ax2.plot(array_mu_sigma_T045_NT[:, 0], array_mu_sigma_T045_NT[:, 1], color = 'cornflowerblue', label = "T = 0.45, P = 2.70 NT" ) 
ax2.fill_between(array_mu_sigma_T045_NT[:, 0], array_mu_sigma_T045_NT[:, 1] + error_T045_NT, array_mu_sigma_T045_NT[:, 1] - error_T045_NT, color = 'lightsteelblue',  linestyle = 'dashdot', edgecolor = 'slategrey')

ax2.plot(array_mu_sigma_T044[:, 0], array_mu_sigma_T044[:, 1], color = 'tab:blue', label = "T = 0.44, P = 2.93" )
ax2.fill_between(array_mu_sigma_T044[:, 0], array_mu_sigma_T044[:, 1] + error_T044, array_mu_sigma_T044[:, 1] - error_T044, color = 'lavender', linestyle = 'dashdot', edgecolor = 'slategrey')

ax2.legend()
ax2.set_xlabel('time (#iterations * Δt)')
ax2.set_ylabel(r'$\langle r^2(t) \rangle$', fontsize = 14) #r' ' per scrivere in latex tra gli apici
ax2.set_xscale('log')
ax2.set_yscale('log')

ax2.grid(linestyle = '--', linewidth = 0.5)

plt.show()

