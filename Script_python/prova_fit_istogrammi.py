import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats

predizioni_T050 = np.load('predictions_k20_e30_bs16_cpMax_T050.npy')

fig = plt.figure()

#fig.suptitle('Prediction on the single conf, DGCNN Trained on all conf\n Density = True, t = 199', fontsize = 14, c='darkgrey')
ax1 = fig.add_subplot(1, 1, 1)

#n, bins_T050, patches = ax1.hist(predizioni_T050[:,1], bins = 'auto',  color = 'orange',  ec = 'skyblue')

#np.where(predizioni_T050[:,1] < 4) # trova gli indici degli elementi nella colonna [:,1] che hanno valore < 4
nuove_predizioni_T050_t199 = np.delete(predizioni_T050[:,1], np.argwhere(predizioni_T050[:,1] < 4.05 ))

n, bins_T050_t199_nuova, patches = ax1.hist(nuove_predizioni_T050_t199, bins = 'auto',  color = 'orange',  ec = 'skyblue', density = True)

#n, bins_T050 = np.histogram(predizioni_T050[:,1], bins = 'auto')
#mu_T050, sigma_T050 = sp.stats.norm.fit(predizioni_T050[:,1])

mu_T050_t199_nuova, sigma_T050_t199_nuova = sp.stats.norm.fit(nuove_predizioni_T050_t199)
#cv_T050 = sigma_T050 / abs(mu_T050)

cv_T050_t199_nuova = sigma_T050_t199_nuova / abs(mu_T050_t199_nuova)
#best_fit_line_T050 = sp.stats.norm.pdf(bins_T050, mu_T050, sigma_T050) # (, media, varianza)
#best_fit_line_T050 = sp.stats.norm.pdf(bins_T050, 4.3, sigma_T050)
    
best_fit_line_T050_t199_nuova = sp.stats.norm.pdf(bins_T050_t199_nuova, mu_T050_t199_nuova, sigma_T050_t199_nuova) # (, media, varianza)

#ax1.set_title(f'T = 0.50 P = 1.55')
#ax1.text(0.1, 0.8, f'μ = {mu_T050:.4f}\nσ = {sigma_T050:.4f}\ncv = {cv_T050:.3f}', fontsize = 10,  transform = ax1.transAxes)

#ax1.plot(bins_T050, best_fit_line_T050)

ax1.plot(bins_T050_t199_nuova, best_fit_line_T050_t199_nuova)
#ax1.plot(predizioni_T050[:,1], bins)

#plt.hist(predizioni_T050[:,1], bins = range(4, 5), density = True)
#plt.xlim(4,4.6)

#freq_soglia = 5
#
#n[np.where(n <= freq_soglia)] = 0
#
#width = 0.7 * (bins_T050[1] - bins_T050[0])
#center = (bins_T050[:-1] + bins_T050[1:]) / 2
#plt.bar(center, n, align='center', width=width)

plt.show()



