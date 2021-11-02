import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats

predizioni_T050 = np.load('predictions_k20_e30_bs16_cpMax_T050.npy')

low_threshold = 4.07

fig = plt.figure()

fig.suptitle(f'T = 0.50, P = 1.55\nexcluding values < {low_threshold}', fontsize = 14, c='darkgrey')
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
n, bins_T050, patches = ax1.hist(predizioni_T050[:,1], bins = 'auto',  color = 'orange',  ec = 'skyblue', density = True)

#np.where(predizioni_T050[:,1] < low_threshold)  trova gli indici degli elementi nella colonna [:,1] che hanno valore < low_threshold
nuove_predizioni_T050_t199 = np.delete(predizioni_T050[:,1], np.argwhere(predizioni_T050[:,1] < low_threshold ))
n, bins_T050_t199_nuova, patches = ax2.hist(nuove_predizioni_T050_t199, bins = 'auto',  color = 'orange',  ec = 'skyblue', density = True)

#n, bins_T050 = np.histogram(predizioni_T050[:,1], bins = 'auto')
mu_T050, sigma_T050 = sp.stats.norm.fit(predizioni_T050[:,1])

mu_T050_t199_nuova, sigma_T050_t199_nuova = sp.stats.norm.fit(nuove_predizioni_T050_t199)
cv_T050 = sigma_T050 / abs(mu_T050)

cv_T050_t199_nuova = sigma_T050_t199_nuova / abs(mu_T050_t199_nuova)
best_fit_line_T050 = sp.stats.norm.pdf(bins_T050, mu_T050, sigma_T050) # (, media, varianza)

best_fit_line_T050_t199_nuova = sp.stats.norm.pdf(bins_T050_t199_nuova, mu_T050_t199_nuova, sigma_T050_t199_nuova)

#ax1.set_title(f'T = 0.50 P = 1.55')

ax1.text(0.1, 0.8, f'μ = {mu_T050:.4f}\nσ = {sigma_T050:.4f}\ncv = {cv_T050:.3f}', fontsize = 10,  transform = ax1.transAxes)
ax2.text(0.1, 0.8, f'μ_nuova = {mu_T050_t199_nuova:.4f}\nσ_nuova = {sigma_T050_t199_nuova:.4f}\ncv_nuova = {cv_T050_t199_nuova:.3f}', fontsize = 10,  transform = ax2.transAxes)

ax1.plot(bins_T050, best_fit_line_T050)
ax2.plot(bins_T050_t199_nuova, best_fit_line_T050_t199_nuova)

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
plt.cla()


