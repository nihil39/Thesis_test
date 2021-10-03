import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.ticker import AutoMinorLocator
#from matplotlib import gridspec

predizioni = np.load('predictions.npy')
test_y = np.load('test.y.npy')

differenza = np.subtract(test_y, predizioni)


# parametri gaussiana
mu = 0
sigma = 0.045

# plt.hist(predizioni)

# n: is the number of counts in each bin of the histogram
# bins: is the left hand edge of each bin
# patches is the individual patches used to create the histogram, e.g a collection of rectangles
# algoritmi per scegliere come calcolare i bin: 'auto', 'sturges', 'fd', 'doane', 'scott', 'rice' or 'sqrt'
fig = plt.figure(figsize=(16,6))

n , bins, patches = plt.hist(differenza[:,2], bins = 'auto', color = 'crimson', ec = 'skyblue', density = True) 

# [:,2] tutto il primo asse, solo il campo 2 del secondo  

# Density parameter, which normalizes bin heights so that the integral of the histogram is 1. The resulting histogram is an approximation of the probability density function.

plt.xticks(bins, rotation = 90) # mette i segni sull'asse x su dove sono i bin, ruota di 90 gradi le scritte
# define minor ticks and draw a grid with them

# Gaussiana

gaussiana = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))

plt.plot(bins, gaussiana, linewidth = 2, color = 'b')

plt.title(f'test.y (label) - predictions\n μ = {mu}, σ = {sigma}', loc = 'center', fontsize = 14)

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




