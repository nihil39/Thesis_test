import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.ticker import AutoMinorLocator
#from matplotlib import gridspec
import scipy as sp
from scipy import stats 

predizioni = np.load('predictions_k20_e30_bs16_cpMax.npy')
test_y = np.load('test_y_k20_e30_bs16_cpMax.npy')

differenza = np.subtract(test_y, predizioni)


# parametri gaussiana
#mu = -0.15
#sigma = 0.23

# plt.hist(predizioni)

# n: is the number of counts in each bin of the histogram
# bins: is the left hand edge of each bin
# patches is the individual patches used to create the histogram, e.g a collection of rectangles
# algoritmi per scegliere come calcolare i bin: 'auto', 'sturges', 'fd', 'doane', 'scott', 'rice' or 'sqrt'
#fig = plt.figure(figsize=(16,6))

fig = plt.figure()
fig.suptitle('test_y (label) - predictions, tutte le configurazioni, Density = True', fontsize = 14, c='blue')

#fig = plt.figure()
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

n , bins1, patches = ax1.hist(differenza[:,0], bins = 'auto', color = 'crimson', ec = 'skyblue', density = True) 
n , bins2, patches = ax2.hist(differenza[:,1], bins = 'auto', color = 'orange', ec = 'skyblue', density = True) 
n , bins3, patches = ax3.hist(differenza[:,2], bins = 'auto', color = 'green', ec = 'skyblue', density = True) 


# [:,2] tutto il primo asse, solo il campo 2 del secondo  

# Density parameter, which normalizes bin heights so that the integral of the histogram is 1. The resulting histogram is an approximation of the probability density function.


mu1, sigma1 = sp.stats.norm.fit(differenza[:,0])
cv1 = sigma1 / abs(mu1) 

mu2, sigma2 = sp.stats.norm.fit(differenza[:,1])
cv2 = sigma2 / abs(mu2) 

mu3, sigma3 = sp.stats.norm.fit(differenza[:,2])
cv3 = sigma3 / abs(mu3) 


best_fit_line1 = sp.stats.norm.pdf(bins1, mu1, sigma1)
best_fit_line2 = sp.stats.norm.pdf(bins2, mu2, sigma1)
best_fit_line3 = sp.stats.norm.pdf(bins3, mu3, sigma1)

ax1.set_title(f't = 0')
ax1.text(0.1, 0.8, f'μ = {mu1:.9f}\nσ = {sigma1:.9f}\ncv = {cv1:.3f} ', fontsize = 10,  transform = ax1.transAxes)

ax2.set_title(f't  = 100')
ax2.text(0.1, 0.8, f'μ = {mu2:.9f}\nσ = {sigma2:.9f}\ncv = {cv2:.3f} ', fontsize = 10,  transform = ax2.transAxes)

ax3.set_title(f't = 199')
ax3.text(0.1, 0.8, f'μ = {mu3:.9f}\nσ = {sigma3:.9f}\ncv = {cv3:.3f} ', fontsize = 10,  transform = ax3.transAxes)

ax1.plot(bins1, best_fit_line1)
ax2.plot(bins2, best_fit_line2)
ax3.plot(bins3, best_fit_line3)

#plt.xticks(bins, rotation = 90) # mette i segni sull'asse x su dove sono i bin, ruota di 90 gradi le scritte
# define minor ticks and draw a grid with them

# Gaussiana

#gaussiana = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     #np.exp(-0.5 * (1 / sigma * (bins - mu))**2))

#plt.plot(bins, gaussiana, linewidth = 2, color = 'b')


#plt.title(f'test.y (label) - predictions\n μ = {mu}, σ = {sigma}', loc = 'center', fontsize = 14)

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




