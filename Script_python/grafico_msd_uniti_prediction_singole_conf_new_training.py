import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats

#RMS e massimo della distribuzione:w

array_mu_sigma_T044 = np.load('array_mu_sigma_T044.npy')
array_mu_sigma_T047 = np.load('array_mu_sigma_T047.npy')
array_mu_sigma_T050 = np.load('array_mu_sigma_T050.npy')
array_mu_sigma_T056 = np.load('array_mu_sigma_T056.npy')
# [tempi, (0: timestep, 1: media partA, 2: sigma partA, 3: media partB, 4: sigma partB]

error_T044 = array_mu_sigma_T044[:, 2] 
error_T047 = array_mu_sigma_T047[:, 2] 
error_T050 = array_mu_sigma_T050[:, 2] 
error_T056 = array_mu_sigma_T056[:, 2] 

predizioni_msd_T044_T044 = np.load('../DGCNN/New_Training/T044_Training/predictions_k20_e30_bs16_cpMax_T044_Training_T044_test.npy')
predizioni_msd_T047_T047 = np.load('../DGCNN/New_Training/T047_Training/predictions_k20_e30_bs16_cpMax_T047_Training_T047_test.npy')
predizioni_msd_T050_T050 = np.load('../DGCNN/New_Training/T050_Training/predictions_k20_e30_bs16_cpMax_T050_Training_T050_test.npy')
predizioni_msd_T056_T056 = np.load('../DGCNN/New_Training/T056_Training/predictions_k20_e30_bs16_cpMax_T056_Training_T056_test.npy')

# [:, 0: tempo t=100, 1: tempo t=199]
#nuove_predizioni_T050_t100 = np.delete(predizioni_msd_T050_all_conf[:,0], np.argwhere(predizioni_msd_T050_all_conf[:,0] < 0.0597 ))
#nuove_predizioni_T050_t199 = np.delete(predizioni_msd_T050_all_conf[:,1], np.argwhere(predizioni_msd_T050_all_conf[:,1] < 4.07))

fig = plt.figure()
fig.suptitle('MSD with shaded error region\n"self" predictions on the training conf', fontsize = 14, c = 'darkgrey')

ax2 = fig.add_subplot(1, 1, 1)

ax2.plot(array_mu_sigma_T044[:, 0], array_mu_sigma_T044[:, 1], color = 'tab:blue', label = "T = 0.44, P = 2.93" ) #ax1.scatter per lo scatter plot
ax2.fill_between(array_mu_sigma_T044[:, 0], array_mu_sigma_T044[:, 1] + error_T044, array_mu_sigma_T044[:, 1] - error_T044, color = 'lavender', linestyle = 'dashdot', edgecolor = 'slategrey') #ax1.scatter per lo scatter plot

ax2.plot(array_mu_sigma_T047[:, 0], array_mu_sigma_T047[:, 1], color = 'mediumseagreen', label = "T = 0.47, P = 2.24" ) #ax1.scatter per lo scatter plot
ax2.fill_between(array_mu_sigma_T047[:, 0], array_mu_sigma_T047[:, 1] + error_T047, array_mu_sigma_T047[:, 1] - error_T047, color = 'mediumspringgreen',linestyle = 'dashdot',  edgecolor = 'slategrey') #ax1.scatter per lo scatter plot

ax2.plot(array_mu_sigma_T050[:, 0], array_mu_sigma_T050[:, 1], color = 'orange', label = "T = 0.50, P = 1.55" ) #ax1.scatter per lo scatter plot
ax2.fill_between(array_mu_sigma_T050[:, 0], array_mu_sigma_T050[:, 1] + error_T050, array_mu_sigma_T050[:, 1] - error_T050, color = 'wheat',linestyle = 'dashdot', edgecolor = 'slategrey') #ax1.scatter per lo scatter plot

ax2.plot(array_mu_sigma_T056[:, 0], array_mu_sigma_T056[:, 1], color = 'tab:red', label = "T = 0.56, P = 0.17" ) #ax1.scatter per lo scatter plot
ax2.fill_between(array_mu_sigma_T056[:, 0], array_mu_sigma_T056[:, 1] + error_T056, array_mu_sigma_T056[:, 1] - error_T056, color = 'indianred',linestyle = 'dashdot', edgecolor = 'slategrey') #ax1.scatter per lo scatter plot

#t = 100
mu_T044_T044_t_100, sigma_T044_T044_t_100 = sp.stats.norm.fit(predizioni_msd_T044_T044[:,0])
mu_T047_T047_t_100, sigma_T047_T047_t_100 = sp.stats.norm.fit(predizioni_msd_T047_T047[:,0])

mu_T050_T050_t_100, sigma_T050_T050_t_100 = sp.stats.norm.fit(predizioni_msd_T050_T050[:,0])
#mu_T050_t_100_nuova, sigma_T050_t_100_nuova = sp.stats.norm.fit(nuove_predizioni_T050_t100)

mu_T056_T056_t_100, sigma_T056_T056_t_100 = sp.stats.norm.fit(predizioni_msd_T056_T056[:,0])

#t = 199
mu_T044_T044_t_199, sigma_T044_T044_t_199 = sp.stats.norm.fit(predizioni_msd_T044_T044[:,1])
mu_T047_T047_t_199, sigma_T047_T047_t_199 = sp.stats.norm.fit(predizioni_msd_T047_T047[:,1])

mu_T050_T050_t_199, sigma_T050_T050_t_199 = sp.stats.norm.fit(predizioni_msd_T050_T050[:,1])
#mu_T050_t_199_nuova, sigma_T050_t_199_nuova = sp.stats.norm.fit(nuove_predizioni_T050_t199)

mu_T056_T056_t_199, sigma_T056_T056_t_199 = sp.stats.norm.fit(predizioni_msd_T056_T056[:,1])

#T = 0.44, P = 2.93
#ax2.plot(1.63 , mu_T044_t_100, color = 'tab:blue', marker = 'o') 
#ax2.plot(500  , mu_T044_t_199, color = 'tab:blue', marker = 'o')
ax2.errorbar(1.63,  mu_T044_T044_t_100, yerr = sigma_T044_T044_t_100, fmt='o', color = 'tab:blue')
ax2.errorbar(500,  mu_T044_T044_t_199, yerr = sigma_T044_T044_t_199, fmt='o',color = 'tab:blue')

###
#
##T = 0.47, P = 2.24
#
#ax2.plot(1.63 ,  mu_T047_t_100,  color = 'mediumseagreen', marker = 's' )
#ax2.plot(500  , mu_T047_t_199,  color = 'mediumseagreen', marker = 's' )

ax2.errorbar(1.63,  mu_T047_T047_t_100, yerr = sigma_T047_T047_t_100, fmt = 's', color = 'mediumseagreen')
ax2.errorbar(500,  mu_T047_T047_t_199, yerr = sigma_T047_T047_t_199, fmt = 's', color = 'mediumseagreen')

##
###T = 0.50, P = 1.55
##
#ax2.plot(1.63 ,  mu_T050_t_100,  color = 'orange', marker = 'p')
#ax2.plot(500  , mu_T050_t_199, color = 'orange', marker = 'p' )

ax2.errorbar(1.63,  mu_T050_T050_t_100, yerr = sigma_T050_T050_t_100, fmt = 'p',color = 'orange')
#ax2.errorbar(500,  mu_T050_T050_t_199, yerr = sigma_T050_T050_t_199, fmt = 'p',color = 'orange')

n_T050_T050_t199, bins_T050_T050_t199 = np.histogram(predizioni_msd_T050_T050[:,1], bins = 'auto', density = True) # t = 199 [:,1]

bin_max = np.where(n_T050_T050_t199 == n_T050_T050_t199.max())
#print(bins_T050_T050_t199[bin_max][0])

ax2.errorbar(500, bins_T050_T050_t199[bin_max][0], yerr = sigma_T050_T050_t_199, fmt = 'p',color = 'orange')

#ax2.errorbar(1.63, mu_T050_T050_t_100_nuova, yerr = sigma_T050_T050_t_100_nuova, fmt = 'p',color = 'orange')
#ax2.errorbar(500,  mu_T050_T050_t_199_nuova, yerr = sigma_T050_T050_t_199_nuova, fmt = 'p',color = 'orange')


##
###T = 0.56, P = 0.17
##
#ax2.plot(1.63 , mu_T056_t_100, color = 'tab:red', marker = '^')
#ax2.plot(500  , mu_T056_t_199, color = 'tab:red', marker = '^' )

ax2.errorbar(1.63,  mu_T056_T056_t_100, yerr = sigma_T056_T056_t_100, fmt='^', color = 'tab:red')
ax2.errorbar(500,  mu_T056_T056_t_199, yerr = sigma_T056_T056_t_199, fmt='^', color = 'tab:red')

ax2.legend()
ax2.set_xlabel('time (#iterations * Δt)')
ax2.set_ylabel(r'$\langle r^2(t) \rangle$', fontsize = 14) #r' ' per scrivere in latex tra gli apici
ax2.set_xscale('log')
ax2.set_yscale('log')

ax2.grid(linestyle = '--', linewidth = 0.5)

plt.show()

