import numpy as np
import matplotlib.pyplot as plt

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


#n , bins, patches = ax1.hist(msd_T044[:,0,1], bins = 'auto', color = 'crimson', ec = 'skyblue', density = True)
#n , bins, patches = ax3.hist(msd_T044[:,100,1], bins = 'auto', color = 'crimson', ec = 'skyblue', density = True)
#n , bins, patches = ax5.hist(msd_T044[:,199,1], bins = 'auto', color = 'crimson', ec = 'skyblue', density = True)
#
#n , bins, patches = ax2.hist(msd_T047[:,0,1], bins = 'auto', color = 'green', ec = 'skyblue', density = True)
#n , bins, patches = ax4.hist(msd_T047[:,100,1], bins = 'auto', color = 'green', ec = 'skyblue', density = True)
#n , bins, patches = ax6.hist(msd_T047[:,199,1], bins = 'auto', color = 'green', ec = 'skyblue', density = True)

n , bins, patches = ax1.hist(msd_T050[:,0,1], bins = 'auto', color = 'crimson', ec = 'skyblue', density = True)
n , bins, patches = ax3.hist(msd_T050[:,100,1], bins = 'auto', color = 'crimson', ec = 'skyblue', density = True)
n , bins, patches = ax5.hist(msd_T050[:,199,1], bins = 'auto', color = 'crimson', ec = 'skyblue', density = True)

n , bins, patches = ax2.hist(msd_T056[:,0,1], bins = 'auto', color = 'green', ec = 'skyblue', density = True)
n , bins, patches = ax4.hist(msd_T056[:,100,1], bins = 'auto', color = 'green', ec = 'skyblue', density = True)
n , bins, patches = ax6.hist(msd_T056[:,199,1], bins = 'auto', color = 'green', ec = 'skyblue', density = True)

ax1.set_title('T = 0.50 P = 1.55 \n t = 0')
ax2.set_title('T = 0.56 P = 0.17 \n t = 0')

ax3.set_title('t = 100')
ax4.set_title('t = 100')

ax5.set_title('t = 199')
ax6.set_title('t = 199')


#ax1.xticks(bins, rotation = 90)
#ax2.xticks(bins, rotation = 90)
#ax3.xticks(bins, rotation = 90)

plt.show()


