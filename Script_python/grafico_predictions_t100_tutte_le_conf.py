import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats

predizioni_T044 = np.load('predictions_k20_e30_bs16_cpMax_T044.npy')
predizioni_T047 = np.load('predictions_k20_e30_bs16_cpMax_T047.npy')
predizioni_T050 = np.load('predictions_k20_e30_bs16_cpMax_T050.npy')

T050_t100_low_threshold = 0.0595
nuove_predizioni_T050_t100 = np.delete(predizioni_T050[:,0], np.argwhere(predizioni_T050[:,0] < T050_t100_low_threshold) )

predizioni_T056 = np.load('predictions_k20_e30_bs16_cpMax_T056.npy')

fig = plt.figure()
fig.suptitle('Prediction on the single conf, DGCNN Trained on all conf\n Density = True, t = 100', fontsize = 14, c='darkgrey')

ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

n, bins_T044, patches = ax1.hist(predizioni_T044[:,0], bins = 'auto', color = 'tab:blue', ec = 'skyblue', density = True)
n, bins_T047, patches = ax2.hist(predizioni_T047[:,0], bins = 'auto', color = 'mediumseagreen', ec = 'skyblue', density = True)

#n, bins_T050, patches = ax3.hist(predizioni_T050[:,0], bins = 'auto', color = 'orange',  ec = 'skyblue', density = True)
n, bins_T050_t100_nuova, patches = ax3.hist(nuove_predizioni_T050_t100, bins = 'auto',  color = 'orange',  ec = 'skyblue', density = True)

n, bins_T056, patches = ax4.hist(predizioni_T056[:,0], bins = 'auto', color = 'tab:red', ec = 'skyblue', density = True)

mu_T044, sigma_T044 = sp.stats.norm.fit(predizioni_T044[:,0])
mu_T047, sigma_T047 = sp.stats.norm.fit(predizioni_T047[:,0])

mu_T050, sigma_T050 = sp.stats.norm.fit(predizioni_T050[:,0])
mu_T050_t100_nuova, sigma_T050_t100_nuova = sp.stats.norm.fit(nuove_predizioni_T050_t100)

mu_T056, sigma_T056 = sp.stats.norm.fit(predizioni_T056[:,0])

cv_T044 = sigma_T044 / abs(mu_T044)
cv_T047 = sigma_T047 / abs(mu_T047)

cv_T050 = sigma_T050 / abs(mu_T050)
cv_T050_t100_nuova = sigma_T050_t100_nuova / abs(mu_T050_t100_nuova)

cv_T056 = sigma_T056 / abs(mu_T056)

best_fit_line_T044 = sp.stats.norm.pdf(bins_T044, mu_T044, sigma_T044)
best_fit_line_T047 = sp.stats.norm.pdf(bins_T047, mu_T047, sigma_T047)
#best_fit_line_T050 = sp.stats.norm.pdf(bins_T050, mu_T050, sigma_T050)
best_fit_line_T050_t100_nuova = sp.stats.norm.pdf(bins_T050_t100_nuova, mu_T050_t100_nuova, sigma_T050_t100_nuova) 

best_fit_line_T056 = sp.stats.norm.pdf(bins_T056, mu_T056, sigma_T056)

ax1.set_title(f'T = 0.44 P = 2.93')
ax1.text(0.1, 0.8, f'μ = {mu_T044:.4f}\nσ = {sigma_T044:.4f}\ncv = {cv_T044:.3f} ', fontsize = 10,  transform = ax1.transAxes)

ax2.set_title(f'T = 0.47 P = 2.24')
ax2.text(0.1, 0.8, f'μ = {mu_T047:.4f}\nσ = {sigma_T047:.4f}\ncv = {cv_T047:.3f}', fontsize = 10,  transform = ax2.transAxes)

#ax3.set_title(f'T = 0.50 P = 1.55')
#ax3.text(0.1, 0.8, f'μ = {mu_T050:.4f}\nσ = {sigma_T050:.4f}\ncv = {cv_T050:.3f}', fontsize = 10,  transform = ax3.transAxes)

ax3.set_title(f'T = 0.50 P = 1.55')
ax3.text(0.1, 0.8, f'μ = {mu_T050_t100_nuova:.4f}\nσ = {sigma_T050_t100_nuova:.4f}\ncv = {cv_T050_t100_nuova:.3f}', fontsize = 10,  transform = ax3.transAxes)

ax4.set_title(f'T = 0.56 P = 0.17')
ax4.text(0.1, 0.8, f'μ = {mu_T056:.4f}\nσ = {sigma_T056:.4f}\ncv = {cv_T056:.3f}', fontsize = 10,  transform = ax4.transAxes)

ax1.plot(bins_T044, best_fit_line_T044)
ax2.plot(bins_T047, best_fit_line_T047)
#ax3.plot(bins_T050, best_fit_line_T050)
ax3.plot(bins_T050_t100_nuova, best_fit_line_T050_t100_nuova)

ax4.plot(bins_T056, best_fit_line_T056)

plt.show()
plt.cla()
