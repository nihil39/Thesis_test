import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats

#from matplotlib.ticker import AutoMinorLocator
#from matplotlib import gridspec

predizioni_T047_T047 = np.load('predictions_k20_e30_bs16_cpMax_T047_Training_T047_test.npy')
test_y_T047_T047 = np.load('test_y_k20_e30_bs_16_cpMax_T047_Training_T047_test.npy')

differenza_T047_T047 = np.subtract(test_y_T047_T047, predizioni_T047_T047)

fig = plt.figure()

ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

fig.suptitle('Trained on T = 0.47, P = 2.24 conf, predictions on the same conf\ntest_y (label) - predictions, Density = True', fontsize = 14, c='darkgrey')

# parametri gaussiana
#mu = 0
#sigma = 0.045

# plt.hist(predizioni)

# n: is the number of counts in each bin of the histogram
# bins: is the left hand edge of each bin
# patches is the individual patches used to create the histogram, e.g a collection of rectangles
# algoritmi per scegliere come calcolare i bin: 'auto', 'sturges', 'fd', 'doane', 'scott', 'rice' or 'sqrt'

#fig = plt.figure(figsize=(16,6))

n, bins_T047_T047_t100, patches = ax1.hist(differenza_T047_T047[:,0], bins = 'auto', color = 'xkcd:dusty red', ec = 'skyblue', density = True) # t = 100 [:,0]
n, bins_T047_T047_t199, patches = ax2.hist(differenza_T047_T047[:,1], bins = 'auto', color = 'xkcd:algae', ec = 'skyblue', density = True) # t = 199 [:,1]

# [:,2] tutto il primo asse, solo il campo 1 del secondo. il secondo campo sono i due tempi a cui è stato predetto il msd 

# Density parameter, which normalizes bin heights so that the integral of the histogram is 1. The resulting histogram is an approximation of the probability density function.

#plt.xticks(bins, rotation = 90) # mette i segni sull'asse x su dove sono i bin, ruota di 90 gradi le scritte
# define minor ticks and draw a grid with them

# Gaussiana

#gaussiana = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
#     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))

mu_T047_T047_t100, sigma_T047_T047_t100 = sp.stats.norm.fit(differenza_T047_T047[:,0])
mu_T047_T047_t199, sigma_T047_T047_t199 = sp.stats.norm.fit(differenza_T047_T047[:,1])

#rv = sp.stats.crystalball(-0.01, 3)

#mu_T047_T047_t100, sigma_T047_T047_t100, = sp.stats.crystalball.fit(differenza_T047_T047[:,0])

cv_T047_T047_t100 = sigma_T047_T047_t100 / abs(mu_T047_T047_t100) 
cv_T047_T047_t199 = sigma_T047_T047_t199 / abs(mu_T047_T047_t199) 

#crystalball_fit_line_T047_T047_t100 = sp.stats.crystalball.pdf(bins_T047_T047_t100, mu_T047_T047_t100, sigma_T047_T047_t100)

gaussian_fit_line_T047_T047_t100 = sp.stats.norm.pdf(bins_T047_T047_t100, mu_T047_T047_t100, sigma_T047_T047_t100)

gaussian_fit_line_T047_T047_t199 = sp.stats.norm.pdf(bins_T047_T047_t199, mu_T047_T047_t199, sigma_T047_T047_t199)

ax1.set_title(f't = 100')
ax1.text(0.1, 0.8, f'μ = {mu_T047_T047_t100:.4f}\nσ = {sigma_T047_T047_t100:.4f}\ncv = {cv_T047_T047_t100:.3f}', fontsize = 10,  transform = ax1.transAxes)

ax2.set_title(f't = 199')
ax2.text(0.1, 0.8, f'μ = {mu_T047_T047_t199:.4f}\nσ = {sigma_T047_T047_t199:.4f}\ncv = {cv_T047_T047_t199:.3f}', fontsize = 10,  transform = ax2.transAxes)

ax1.plot(bins_T047_T047_t100, gaussian_fit_line_T047_T047_t100)

#ax1.plot(bins_T047_T047_t100, sp.stats.crystalball.pdf(bins_T047_T047_t100, -0.01,3,0))

ax2.plot(bins_T047_T047_t199, gaussian_fit_line_T047_T047_t199)

#plt.plot(bins, gaussiana, linewidth = 2, color = 'b')
#plt.title(f'test.y (label) - predictions\n μ = {mu1}, σ = {sigma1}', loc = 'center', fontsize = 14, c='black')

plt.show()

#https://stackoverflow.com/questions/11315641/python-plotting-a-histogram-with-a-function-line-on-top

#minor_locator = AutoMinorLocator(2)
#plt.gca().xaxis.set_minor_locator(minor_locator)
#plt.grid(which='minor', color='white', lw = 0.5)

# x ticks
#xticks = [(bins[idx+1] + value)/2 for idx, value in enumerate(bins[:-1])]

#xticks_labels = [ "{:.2f}\n-\n{:.2f}".format(value, bins[idx+1]) for idx, value in enumerate(bins[:-1])]

#plt.xticks(xticks, labels = xticks_labels)
#print(f'numero di occorrenze ', n)
#print(f'bins', bins)
#print(patches)

#plt.hist(test_y)
