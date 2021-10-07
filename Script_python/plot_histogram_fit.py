import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats

msd_T044 = np.load('msd_completo_T044_P293.npy')
msd_T047 = np.load('msd_completo_T047_P224.npy')
msd_T050 = np.load('msd_completo_T050_P155.npy')
msd_T056 = np.load('msd_completo_T056_P017.npy')

fig = plt.figure()

ax1 = fig.add_subplot(3, 2, 1)
ax2 = fig.add_subplot(3, 2, 2)
ax3 = fig.add_subplot(3, 2, 3)
ax4 = fig.add_subplot(3, 2, 4)
ax5 = fig.add_subplot(3, 2, 5)
ax6 = fig.add_subplot(3, 2, 6)


fig.suptitle('MSD histogram, density = True', fontsize = 14, c='blue')


#If density = True, draw and return a probability density: each bin will display the bin's raw count divided by the total number of counts and the bin width (density = counts / (sum(counts) * np.diff(bins))), so that the area under the histogram  integrates to 1 (np.sum(density * np.diff(bins)) == 1). get the probability to be in a bin by multiplying the height by the width of the bin
# 
# If stacked is also True, the sum of the histograms is normalized to 1.


#n, bins1, patches = ax1.hist(msd_T044[:,0,1], bins = 'auto', color = 'crimson', ec = 'skyblue', density = True)
#n, bins3, patches = ax3.hist(msd_T044[:,100,1], bins = 'auto', color = 'crimson', ec = 'skyblue', density = True)
#n, bins5, patches = ax5.hist(msd_T044[:,199,1], bins = 'auto', color = 'crimson', ec = 'skyblue', density = True)
#
#n, bins2, patches = ax2.hist(msd_T047[:,0,1], bins = 'auto', color = 'green', ec = 'skyblue', density = True)
#n, bins4, patches = ax4.hist(msd_T047[:,100,1], bins = 'auto', color = 'green', ec = 'skyblue', density = True)
#n, bins6, patches = ax6.hist(msd_T047[:,199,1], bins = 'auto', color = 'green', ec = 'skyblue', density = True)

n , bins1, patches = ax1.hist(msd_T050[:,0,1], bins = 'auto', color = 'crimson', ec = 'skyblue', density = True)
n , bins3, patches = ax3.hist(msd_T050[:,100,1], bins = 'auto', color = 'crimson', ec = 'skyblue', density = True)
n , bins5, patches = ax5.hist(msd_T050[:,199,1], bins = 'auto', color = 'crimson', ec = 'skyblue', density = True)

n , bins2, patches = ax2.hist(msd_T056[:,0,1], bins = 'auto', color = 'green', ec = 'skyblue', density = True)
n , bins4, patches = ax4.hist(msd_T056[:,100,1], bins = 'auto', color = 'green', ec = 'skyblue', density = True)
n , bins6, patches = ax6.hist(msd_T056[:,199,1], bins = 'auto', color = 'green', ec = 'skyblue', density = True)

#dispari colonna sinistra
#mu1, sigma1 = sp.stats.norm.fit(msd_T044[:,0,1])
#mu3, sigma3 = sp.stats.norm.fit(msd_T044[:,100,1])
#mu5, sigma5 = sp.stats.norm.fit(msd_T044[:,199,1])

#pari colonna destra
#mu2, sigma2 = sp.stats.norm.fit(msd_T047[:,0,1])
#mu4, sigma4 = sp.stats.norm.fit(msd_T047[:,100,1])
#mu6, sigma6 = sp.stats.norm.fit(msd_T047[:,199,1])


mu1, sigma1 = sp.stats.norm.fit(msd_T050[:,0,1])
mu3, sigma3 = sp.stats.norm.fit(msd_T050[:,100,1])
mu5, sigma5 = sp.stats.norm.fit(msd_T050[:,199,1])

mu2, sigma2 = sp.stats.norm.fit(msd_T056[:,0,1])
mu4, sigma4 = sp.stats.norm.fit(msd_T056[:,100,1])
mu6, sigma6 = sp.stats.norm.fit(msd_T056[:,199,1])


cv1 = sigma1 / abs(mu1) 
cv2 = sigma2 / abs(mu2) 
cv3 = sigma3 / abs(mu3) 
cv4 = sigma4 / abs(mu4) 
cv5 = sigma5 / abs(mu5) 
cv6 = sigma6 / abs(mu6) 

#mu1, sigma1 = sp.stats.norm.fit(msd_T050[:,0,1])
#mu3, sigma3 = sp.stats.norm.fit(msd_T050[:,100,1])
#mu5, sigma5 = sp.stats.norm.fit(msd_T050[:,199,1])
#mu2, sigma2 = sp.stats.norm.fit(msd_T056[:,0,1])
#mu4, sigma4 = sp.stats.norm.fit(msd_T056[:,100,1])
#mu6, sigma6 = sp.stats.norm.fit(msd_T056[:,199,1])

best_fit_line1 = sp.stats.norm.pdf(bins1, mu1, sigma1)
best_fit_line2 = sp.stats.norm.pdf(bins2, mu2, sigma2)
best_fit_line3 = sp.stats.norm.pdf(bins3, mu3, sigma3)
best_fit_line4 = sp.stats.norm.pdf(bins4, mu4, sigma4)
best_fit_line5 = sp.stats.norm.pdf(bins5, mu5, sigma5)
best_fit_line6 = sp.stats.norm.pdf(bins6, mu6, sigma6)

ax1.set_title('T = 0.50 P = 1.55 \n t = 0')
ax1.text(0.1, 0.8, f'mu = {mu1:.9f} \nsigma = {sigma1:.9f}\ncv = {cv1:.3f}  ', fontsize = 10,  transform = ax1.transAxes) #:w

ax2.set_title('T = 0.56 P = 0.17 \n t = 0')
ax2.text(0.1, 0.8, f'mu = {mu2:.9f} \nsigma = {sigma2:.9f}\ncv = {cv2:.3f}  ', fontsize = 10,  transform = ax2.transAxes)

ax3.set_title('t = 100')
ax3.text(0.1, 0.8, f'mu = {mu3:.4f} \nsigma = {sigma3:.4f}\ncv = {cv3:.3f}  ', fontsize = 10,  transform = ax3.transAxes)

ax4.set_title('t = 100')
ax4.text(0.1, 0.8, f'mu = {mu4:.4f} \nsigma = {sigma4:.4f}\ncv = {cv4:.3f}  ', fontsize = 10,  transform = ax4.transAxes)

ax5.set_title('t = 199')
ax5.text(0.1, 0.8, f'mu = {mu5:.4f} \nsigma = {sigma5:.4f}\ncv = {cv5:.3f}  ', fontsize = 10,  transform = ax5.transAxes)

ax6.set_title('t = 199')
ax6.text(0.1, 0.8, f'mu = {mu6:.4f} \nsigma = {sigma6:.4f}\ncv = {cv6:.3f}  ', fontsize = 10,  transform = ax6.transAxes)

ax1.plot(bins1, best_fit_line1)
ax2.plot(bins2, best_fit_line2)
ax3.plot(bins3, best_fit_line3)
ax4.plot(bins4, best_fit_line4)
ax5.plot(bins5, best_fit_line5)
ax6.plot(bins6, best_fit_line6)

#ax1.xticks(bins, rotation = 90)
#ax2.xticks(bins, rotation = 90)
#ax3.xticks(bins, rotation = 90)
#ax1.xticks(bins, rotation = 90)
#ax2.xticks(bins, rotation = 90)
#ax3.xticks(bins, rotation = 90)

plt.show()


