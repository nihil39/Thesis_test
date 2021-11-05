import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats

fig = plt.figure()

tempo = 199

fig.suptitle(f'Distribuzione msd al tempo {tempo}', fontsize = 14, c = 'darkgrey')

ax1 = fig.add_subplot(1, 1, 1)

array_msd_T044 = np.load('msd_completo_T044_P293.npy')

tipo_particelle = 1

n, bins_T044, patches = ax1.hist(array_msd_T044[:, tempo, tipo_particelle], bins = 'auto',  color = 'tab:blue',  ec = 'skyblue', density = True)

mu_T044, sigma_T044 = sp.stats.norm.fit(array_msd_T044[:, tempo, tipo_particelle])

best_fit_line_T044 = sp.stats.norm.pdf(bins_T044, mu_T044, sigma_T044) # (, media, varianza)
ax1.plot(bins_T044, best_fit_line_T044)

plt.show()



