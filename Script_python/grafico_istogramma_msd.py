import numpy as np
import matplotlib.pyplot as plt

array_msd_T044 = np.load('msd_completo_T044_P293.npy')
array_msd_T047 = np.load('msd_completo_T047_P224.npy')
array_msd_T050 = np.load('msd_completo_T050_P155.npy')
array_msd_T056 = np.load('msd_completo_T056_P017.npy') #[configurazioni, (questo campo deve scorrere coi :),  0 = tempi / 1 = msdA / 2 = msdB]

#tempo = 100
msdA = 1
msdB = 2

fig = plt.figure()
#fig.suptitle(f'Istogramma msd, tempo = {tempo}', fontsize = 14, c = 'darkgrey')

fig.suptitle('Distribuzione msd a' + r' $\tau_g$' + '\n' + r'$\tau_g \approx e^{\frac{a}{T-T_0}}$', fontsize = 14, c = 'darkgrey')

ax1 = fig.add_subplot(1, 1, 1)

kwargs_solo_contorno = dict(histtype = 'step', alpha = 0.3,  bins = 'auto', density = True, linewidth = 2)

# step per far graficare solo il contorno dell'istogramma, stepfilled per un istogramma riempito
#n, bins1, patches = ax1.hist(array_msd_T044[:, 199, msdA], color = 'blue', ec = 'tab:blue', **kwargs_solo_contorno, label = "T = 0.44, P = 2.93") 
#n, bins1, patches = ax1.hist(array_msd_T047[:, 159, msdA], color = 'green', ec = 'mediumseagreen', **kwargs_solo_contorno, label = "T = 0.47, P = 2.24")
n, bins1, patches = ax1.hist(array_msd_T050[:, 187, msdA], color = 'orange', ec = 'orange', **kwargs_solo_contorno, label = "T = 0.50, P = 1.55")
n, bins1, patches = ax1.hist(array_msd_T056[:, 156, msdA], color = 'red', ec = 'tab:red', **kwargs_solo_contorno, label = "T = 0.56, P = 0.17")

ax1.legend()
#ax1.set_title(f'T = 0.44 P = 2.93')

plt.show()
