import os
import numpy as np
import matplotlib.pyplot as plt
import imageio

array_msd_T044 = np.load('../msd_completo_T044_P293.npy')
array_msd_T047 = np.load('../msd_completo_T047_P224.npy')
array_msd_T050 = np.load('../msd_completo_T050_P155.npy')
array_msd_T056 = np.load('../msd_completo_T056_P017.npy') #[configurazioni, (questo campo deve scorrere coi :),  0 = tempi / 1 = msdA / 2 = msdB]

#tempo = 100

fig = plt.figure()
fig.suptitle(f'Istogramma msd', fontsize = 14, c = 'darkgrey')

ax1 = fig.add_subplot(1, 1, 1)

kwargs_solo_contorno = dict(histtype = 'step', alpha = 0.3,  bins = 'auto', density = True, linewidth = 2)

msdA = 1
msdB = 2

#lista_tempi = [100, 105, 110]

filenames = []

for time in range(0,199):

    ax1.hist(array_msd_T044[:, time, msdA], color = 'tab:blue', ec = 'tab:blue', **kwargs_solo_contorno, label = "T = 0.44, P = 2.93") 
    ax1.hist(array_msd_T047[:, time, msdA], color = 'mediumseagreen', ec = 'mediumseagreen', **kwargs_solo_contorno, label = "T = 0.47, P = 2.24") 
    ax1.hist(array_msd_T050[:, time, msdA], color = 'orange', ec = 'orange', **kwargs_solo_contorno, label = "T = 0.50, P = 1.55") 
    ax1.hist(array_msd_T056[:, time, msdA], color = 'tab:red', ec = 'tab:red', **kwargs_solo_contorno, label = "T = 0.56, P = 0.17") 
    ax1.legend()
    plt.draw()
    filename = f'{time}.png'
    filenames.append(filename)
    fig.savefig(f'./{time}.png')
    plt.cla() # clears axes indexes
#del tempo

# Build GIF
with imageio.get_writer('mygif_tot.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Remove files
for filename in set(filenames):
    os.remove(filename)

#tempo = 150
#ax1.hist(array_msd_T044[:, tempo, msdA], color = 'blue', ec = 'tab:blue', **kwargs_solo_contorno, label = "T = 0.44, P = 2.93") 
#ax1.legend()
##plt.draw()
#fig.savefig(f'./{tempo}.png')

#ax1.set_title(f'T = 0.44 P = 2.93')

#plt.show()
