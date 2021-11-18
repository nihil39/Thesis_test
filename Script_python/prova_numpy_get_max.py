import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats

#from matplotlib.ticker import AutoMinorLocator
#from matplotlib import gridspec

predizioni_Tutte_T050 = np.load('./Tutte_Training_new/predictions_k20_e30_bs16_cpMax_Tutte_Training_T050_test.npy')

fig = plt.figure()

ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

n, bins_Tutte_T050_t100, patches = ax1.hist(predizioni_Tutte_T050[:,0], bins = 'auto', color = 'xkcd:dusty red', ec = 'skyblue', density = True) # t = 100 [:,0]
n, bins_Tutte_T050_t199, patches = ax2.hist(predizioni_Tutte_T050[:,1], bins = 'auto', color = 'xkcd:algae', ec = 'skyblue', density = True) # t = 199 [:,1]

n, bins_Tutte_T050_t100_no_density, patches = ax3.hist(predizioni_Tutte_T050[:,0], bins = 'auto', color = 'xkcd:dusty red', ec = 'skyblue' ) # t = 100 [:,0]
n, bins_Tutte_T050_t199_no_density, patches = ax4.hist(predizioni_Tutte_T050[:,1], bins = 'auto', color = 'xkcd:algae', ec = 'skyblue') # t = 199 [:,1]

n_t100_numpy, bins_Tutte_T050_t100_numpy = np.histogram(predizioni_Tutte_T050[:,0], bins = 'auto', density = True) # t = 100 [:,0]
n_t199_numpy, bins_Tutte_T050_t199_numpy = np.histogram(predizioni_Tutte_T050[:,1], bins = 'auto', density = True) # t = 199 [:,1]

n_t100_numpy_no_density, bins_Tutte_T050_t100_numpy_no_density = np.histogram(predizioni_Tutte_T050[:,0], bins = 'auto') # t = 100 [:,0]
n_t199_numpy_no_density, bins_Tutte_T050_t199_numpy_no_density = np.histogram(predizioni_Tutte_T050[:,1], bins = 'auto') # t = 199 [:,1]


fig.suptitle('Trained on T = 0.50, P = 1.55 conf, predictions on the same conf', fontsize = 14, c='darkgrey')
max_Tutte_T050_t100 = bins_Tutte_T050_t100_numpy[np.where(n_t100_numpy == n_t100_numpy.max())][0]

print('histogram max t=100', max_Tutte_T050_t100)

bin_max = np.where(n_t199_numpy == n_t199_numpy.max())
max_Tutte_T050_t199 = bins_Tutte_T050_t199_numpy[bin_max][0]
print('histogram max', max_Tutte_T050_t199)

# How to calculate the standard deviation from a histogram with numpy
# https://stackoverflow.com/questions/50786699/how-to-calculate-the-standard-deviation-from-a-histogram-python-matplotlib/57400289
mids = 0.5 * (bins_Tutte_T050_t100_numpy[1:] + bins_Tutte_T050_t100_numpy[:-1]) # midpoints of the bins
mean = np.average(mids, weights = n_t100_numpy)
var = np.average((mids - mean)**2, weights = n_t100_numpy)
print('estimated mean t100 ', mean)
print('estimated rms t100 ', np.sqrt(var))

mids = 0.5 * (bins_Tutte_T050_t199_numpy[1:] + bins_Tutte_T050_t199_numpy[:-1]) # midpoints of the bins
mean = np.average(mids, weights = n_t199_numpy)
var = np.average((mids - mean)**2, weights = n_t199_numpy)
print('estimated mean ', mean)
print('estimated rms t199 ', np.sqrt(var))

mids = 0.5 * (bins_Tutte_T050_t100_numpy_no_density[1:] + bins_Tutte_T050_t100_numpy_no_density[:-1]) # midpoints of the bins
mean = np.average(mids, weights = n_t100_numpy_no_density)
var = np.average((mids - mean)**2, weights = n_t100_numpy_no_density)
print('estimated mean t100 ', mean)
print('estimated rms t100 no density ', np.sqrt(var))

mids = 0.5 * (bins_Tutte_T050_t199_numpy_no_density[1:] + bins_Tutte_T050_t199_numpy_no_density[:-1]) # midpoints of the bins
mean = np.average(mids, weights = n_t199_numpy_no_density)
var = np.average((mids - mean)**2, weights = n_t199_numpy_no_density)
print('estimated mean ', mean)
print('estimated rms t199 no density ', np.sqrt(var))

mu_Tutte_T050_t100, sigma_Tutte_T050_t100 = sp.stats.norm.fit(predizioni_Tutte_T050[:,0])
mu_Tutte_T050_t199, sigma_Tutte_T050_t199 = sp.stats.norm.fit(predizioni_Tutte_T050[:,1])

cv_Tutte_T050_t100 = sigma_Tutte_T050_t100 / abs(mu_Tutte_T050_t100) 
cv_Tutte_T050_t199 = sigma_Tutte_T050_t199 / abs(mu_Tutte_T050_t199) 

gaussian_fit_line_Tutte_T050_t100 = sp.stats.norm.pdf(bins_Tutte_T050_t100, mu_Tutte_T050_t100, sigma_Tutte_T050_t100)
gaussian_fit_line_Tutte_T050_t199 = sp.stats.norm.pdf(bins_Tutte_T050_t199, mu_Tutte_T050_t199, sigma_Tutte_T050_t199)


ax1.set_title(f'T = 0.50 t = 100 density')
ax1.text(0.1, 0.8, f'μ = {mu_Tutte_T050_t100:.4f}\nσ = {sigma_Tutte_T050_t100:.4f}\ncv = {cv_Tutte_T050_t100:.3f}', fontsize = 10,  transform = ax1.transAxes)

ax2.set_title(f'T = 0.50 t = 199 density')
ax2.text(0.1, 0.8, f'μ = {mu_Tutte_T050_t199:.4f}\nσ = {sigma_Tutte_T050_t199:.4f}\ncv = {cv_Tutte_T050_t199:.3f}', fontsize = 10,  transform = ax2.transAxes)

ax3.set_title(f'T = 0.56 t = 100')

ax4.set_title(f'T = 0.56 t = 199')

ax1.plot(bins_Tutte_T050_t100, gaussian_fit_line_Tutte_T050_t100)
ax2.plot(bins_Tutte_T050_t199, gaussian_fit_line_Tutte_T050_t199)
#ax3.plot(bins_Tutte_T050_t100_no_density)
#ax4.plot(bins_Tutte_T050_t199_no_density)

plt.show()
