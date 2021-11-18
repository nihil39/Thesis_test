import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats

#RMS e massimo della distribuzione:w

array_mu_sigma_T045 = np.load('array_mu_sigma_T045_NT.npy')
array_mu_sigma_T049 = np.load('array_mu_sigma_T049_NT.npy')
array_mu_sigma_T052 = np.load('array_mu_sigma_T052_NT.npy')
# [tempi, (0: timestep, 1: media partA, 2: sigma partA, 3: media partB, 4: sigma partB]

error_T045 = array_mu_sigma_T045[:, 2] 
error_T049 = array_mu_sigma_T049[:, 2] 
error_T052 = array_mu_sigma_T052[:, 2] 

predizioni_msd_Tutte_T045 = np.load('../DGCNN/New_Training/Tutte_Training_new/predictions_k20_e30_bs16_cpMax_Tutte_Training_T045_test.npy')
predizioni_msd_Tutte_T049 = np.load('../DGCNN/New_Training/Tutte_Training_new/predictions_k20_e30_bs16_cpMax_Tutte_Training_T049_test.npy')
predizioni_msd_Tutte_T052 = np.load('../DGCNN/New_Training/Tutte_Training_new/predictions_k20_e30_bs16_cpMax_Tutte_Training_T052_test.npy')

n_T045_t100, bins_Tutte_T045_t100 = np.histogram(predizioni_msd_Tutte_T045[:,0], bins = 'auto', density = True) # t = 100 [:,0]
n_T045_t199, bins_Tutte_T045_t199 = np.histogram(predizioni_msd_Tutte_T045[:,1], bins = 'auto', density = True) # t = 100 [:,0]

n_T049_t100, bins_Tutte_T049_t100 = np.histogram(predizioni_msd_Tutte_T049[:,0], bins = 'auto', density = True) # t = 199 [:,1]
n_T049_t199, bins_Tutte_T049_t199 = np.histogram(predizioni_msd_Tutte_T049[:,1], bins = 'auto', density = True) # t = 199 [:,1]

n_T052_t100, bins_Tutte_T052_t100 = np.histogram(predizioni_msd_Tutte_T052[:,0], bins = 'auto', density = True)  

max_Tutte_T052_t100 = bins_Tutte_T052_t100[np.where(n_T052_t100 == n_T052_t100.max())][0]

mids = 0.5 * (bins_Tutte_T052_t100[1:] + bins_Tutte_T052_t100[:-1]) # midpoints of the bins
mean = np.average(mids, weights = n_T052_t100)
var = np.average((mids - mean)**2, weights = n_T052_t100)
rms_Tutte_T052_t100 = np.sqrt(var)

#print('max T052 t100', max_Tutte_T052_t100)
#print('rms T052 t100', rms_Tutte_T052_t100)

n_T052_t199, bins_Tutte_T052_t199 = np.histogram(predizioni_msd_Tutte_T052[:,1], bins = 'auto', density = True)  

max_Tutte_T052_t199 = bins_Tutte_T052_t199[np.where(n_T052_t199 == n_T052_t199.max())][0]

mids = 0.5 * (bins_Tutte_T052_t199[1:] + bins_Tutte_T052_t199[:-1]) # midpoints of the bins
mean = np.average(mids, weights = n_T052_t199)
var = np.average((mids - mean)**2, weights = n_T052_t199)
rms_Tutte_T052_t199 = np.sqrt(var)

print('max T052 t199', max_Tutte_T052_t199)
print('rms T052 t199', rms_Tutte_T052_t199)


#bin_max = np.where(n_Tutte_T050_t199 == n_Tutte_T050_t199.max())

# [:, 0: tempo t=100, 1: tempo t=199]
#nuove_predizioni_T050_t100 = np.delete(predizioni_msd_T050_all_conf[:,0], np.argwhere(predizioni_msd_T050_all_conf[:,0] < 0.0597 ))
#nuove_predizioni_T050_t199 = np.delete(predizioni_msd_T050_all_conf[:,1], np.argwhere(predizioni_msd_T050_all_conf[:,1] < 4.07))

fig = plt.figure()
fig.suptitle('MSD with shaded error region\nNN trained on all conf, predictions on the other conf', fontsize = 14, c = 'darkgrey')

ax2 = fig.add_subplot(1, 1, 1)

ax2.plot(array_mu_sigma_T052[:, 0], array_mu_sigma_T052[:, 1], color = 'coral', label = "T = 0.52, P = 1.09 NT" ) 
ax2.fill_between(array_mu_sigma_T052[:, 0], array_mu_sigma_T052[:, 1] + error_T052, array_mu_sigma_T052[:, 1] - error_T052, color = 'mistyrose',linestyle = 'dashdot', edgecolor = 'slategrey') 

ax2.plot(array_mu_sigma_T049[:, 0], array_mu_sigma_T049[:, 1], color = 'gold', label = "T = 0.49, P = 1.78 NT" )
ax2.fill_between(array_mu_sigma_T049[:, 0], array_mu_sigma_T049[:, 1] + error_T049, array_mu_sigma_T049[:, 1] - error_T049, color = 'xkcd:cream',linestyle = 'dashdot', edgecolor = 'slategrey') 

ax2.plot(array_mu_sigma_T045[:, 0], array_mu_sigma_T045[:, 1], color = 'cornflowerblue', label = "T = 0.45, P = 2.70 NT" ) 
ax2.fill_between(array_mu_sigma_T045[:, 0], array_mu_sigma_T045[:, 1] + error_T045, array_mu_sigma_T045[:, 1] - error_T045, color = 'lightsteelblue',  linestyle = 'dashdot', edgecolor = 'slategrey')



#t = 100
mu_Tutte_T045_t_100, sigma_Tutte_T045_t_100 = sp.stats.norm.fit(predizioni_msd_Tutte_T045[:,0])
mu_Tutte_T049_t_100, sigma_Tutte_T049_t_100 = sp.stats.norm.fit(predizioni_msd_Tutte_T049[:,0])
mu_Tutte_T052_t_100, sigma_Tutte_T052_t_100 = sp.stats.norm.fit(predizioni_msd_Tutte_T052[:,0])
#mu_T050_t_100_nuova, sigma_T050_t_100_nuova = sp.stats.norm.fit(nuove_predizioni_T050_t100)


#t = 199
mu_Tutte_T045_t_199, sigma_Tutte_T045_t_199 = sp.stats.norm.fit(predizioni_msd_Tutte_T045[:,1])
mu_Tutte_T049_t_199, sigma_Tutte_T049_t_199 = sp.stats.norm.fit(predizioni_msd_Tutte_T049[:,1])
mu_Tutte_T052_t_199, sigma_Tutte_T052_t_199 = sp.stats.norm.fit(predizioni_msd_Tutte_T052[:,1])
#mu_T050_t_199_nuova, sigma_T050_t_199_nuova = sp.stats.norm.fit(nuove_predizioni_T050_t199)


#T = 0.44, P = 2.93
#ax2.plot(1.63 , mu_T044_t_100, color = 'tab:blue', marker = 'o') 
#ax2.plot(500  , mu_T044_t_199, color = 'tab:blue', marker = 'o')
ax2.errorbar(1.63,  mu_Tutte_T045_t_100, yerr = sigma_Tutte_T045_t_100, fmt='o', color = 'cornflowerblue')
ax2.errorbar(500,   mu_Tutte_T045_t_199, yerr = sigma_Tutte_T045_t_199, fmt='o',color = 'cornflowerblue')

###
#
##T = 0.47, P = 2.24
#
#ax2.plot(1.63 ,  mu_T047_t_100,  color = 'mediumseagreen', marker = 's' )
#ax2.plot(500  , mu_T047_t_199,  color = 'mediumseagreen', marker = 's' )

ax2.errorbar(1.63,  mu_Tutte_T049_t_100, yerr = sigma_Tutte_T049_t_100, fmt = 's', color = 'gold')
ax2.errorbar(500,   mu_Tutte_T049_t_199, yerr = sigma_Tutte_T049_t_199, fmt = 's', color = 'gold')

##
###T = 0.50, P = 1.55
##
#ax2.plot(1.63 ,  mu_T050_t_100,  color = 'orange', marker = 'p')
#ax2.plot(500  , mu_T050_t_199, color = 'orange', marker = 'p' )

#ax2.errorbar(1.63,  mu_Tutte_T052_t_100, yerr = sigma_Tutte_T052_t_100, fmt = 'p',color = 'coral')
#ax2.errorbar(500,   mu_Tutte_T052_t_199, yerr = sigma_Tutte_T052_t_199, fmt = 'p',color = 'coral')

ax2.errorbar(1.63,  max_Tutte_T052_t100, yerr = rms_Tutte_T052_t100, fmt = 'p',color = 'coral')
ax2.errorbar(500,   max_Tutte_T052_t199, yerr = rms_Tutte_T052_t199, fmt = 'p',color = 'coral')


#print(bins_Tutte_T050_t199[bin_max][0])

#ax2.errorbar(500, bins_Tutte_T050_t199[bin_max][0], yerr = sigma_Tutte_T050_t_199, fmt = 'p',color = 'orange')

#ax2.errorbar(1.63, mu_Tutte_T050_t_100_nuova, yerr = sigma_Tutte_T050_t_100_nuova, fmt = 'p',color = 'orange')
#ax2.errorbar(500,  mu_Tutte_T050_t_199_nuova, yerr = sigma_Tutte_T050_t_199_nuova, fmt = 'p',color = 'orange')

ax2.legend()
ax2.set_xlabel('time (#iterations * Δt)')
ax2.set_ylabel(r'$\langle r^2(t) \rangle$', fontsize = 14) #r' ' per scrivere in latex tra gli apici
ax2.set_xscale('log')
ax2.set_yscale('log')

ax2.grid(linestyle = '--', linewidth = 0.5)

plt.show()

