import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats

#from matplotlib.ticker import AutoMinorLocator
#from matplotlib import gridspec

predizioni1 = np.load('predictions_k20_e30_bs16_cpMax_T044_P293.npy')
test_y1 = np.load('test_y_k20_e30_bs16_cpMax_T044_P293.npy')

predizioni2 = np.load('predictions_k20_e30_bs16_cpMax_T047_P224.npy')
test_y2 = np.load('test_y_k20_e30_bs16_cpMax_T047_P224.npy')

predizioni3 = np.load('predictions_k20_e30_bs16_cpMax_T050_P155.npy')
test_y3 = np.load('test_y_k20_e30_bs16_cpMax_T050_P155.npy')

predizioni4 = np.load('predictions_k20_e30_bs16_cpMax_T056_P017.npy')
test_y4 = np.load('test_y_k20_e30_bs16_cpMax_T056_P017.npy')


differenza1 = np.subtract(test_y1, predizioni1)
differenza2 = np.subtract(test_y2, predizioni2)
differenza3 = np.subtract(test_y3, predizioni3)
differenza4 = np.subtract(test_y4, predizioni4)

fig = plt.figure()

ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)


fig.suptitle('test_y (label) - predictions, Density = True', fontsize = 14, c='darkgrey')

# parametri gaussiana
#mu = 0
#sigma = 0.045

# plt.hist(predizioni)

# n: is the number of counts in each bin of the histogram
# bins: is the left hand edge of each bin
# patches is the individual patches used to create the histogram, e.g a collection of rectangles
# algoritmi per scegliere come calcolare i bin: 'auto', 'sturges', 'fd', 'doane', 'scott', 'rice' or 'sqrt'

#fig = plt.figure(figsize=(16,6))
n, bins1, patches = ax1.hist(differenza1[:,2], bins = 'auto', color = 'tab:blue', ec = 'skyblue', density = True)
n, bins2, patches = ax2.hist(differenza2[:,2], bins = 'auto', color = 'mediumseagreen', ec = 'skyblue', density = True)
n, bins3, patches = ax3.hist(differenza3[:,2], bins = 'auto', color = 'orange', ec = 'skyblue', density = True)
n, bins4, patches = ax4.hist(differenza4[:,2], bins = 'auto', color = 'tab:red', ec = 'skyblue', density = True)

# [:,2] tutto il primo asse, solo il campo 2 del secondo. il secondo campo sono i tre tempi a cui è stato predetto il msd 

# Density parameter, which normalizes bin heights so that the integral of the histogram is 1. The resulting histogram is an approximation of the probability density function.

#plt.xticks(bins, rotation = 90) # mette i segni sull'asse x su dove sono i bin, ruota di 90 gradi le scritte
# define minor ticks and draw a grid with them

# Gaussiana

#gaussiana = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
#     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))

mu1, sigma1 = sp.stats.norm.fit(differenza1[:,2])
mu2, sigma2 = sp.stats.norm.fit(differenza2[:,2])
mu3, sigma3 = sp.stats.norm.fit(differenza3[:,2])
mu4, sigma4 = sp.stats.norm.fit(differenza4[:,2])

best_fit_line1 = sp.stats.norm.pdf(bins1, mu1, sigma1, color='black')
best_fit_line2 = sp.stats.norm.pdf(bins2, mu2, sigma2)
best_fit_line3 = sp.stats.norm.pdf(bins3, mu3, sigma3)
best_fit_line4 = sp.stats.norm.pdf(bins4, mu4, sigma4)

ax1.set_title(f'T = 0.44 P = 2.93')
ax1.text(0.1, 0.8, f' μ = {mu1:.4f}\nσ = {sigma1:.4f} ', fontsize = 10,  transform = ax1.transAxes)

ax2.set_title(f'T = 0.47 P = 2.24')
ax2.text(0.1, 0.8, f' μ = {mu2:.4f}\nσ = {sigma2:.4f} ', fontsize = 10,  transform = ax2.transAxes)

ax3.set_title(f'T = 0.50 P = 1.55')
ax3.text(0.1, 0.8, f' μ = {mu3:.4f}\nσ = {sigma3:.4f} ', fontsize = 10,  transform = ax3.transAxes)

ax4.set_title(f'T = 0.56 P = 0.17')
ax4.text(0.1, 0.8, f' μ = {mu4:.4f}\nσ = {sigma4:.4f} ', fontsize = 10,  transform = ax4.transAxes)

ax1.plot(bins1, best_fit_line1)
ax2.plot(bins2, best_fit_line2)
ax3.plot(bins3, best_fit_line3)
ax4.plot(bins4, best_fit_line4)

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




