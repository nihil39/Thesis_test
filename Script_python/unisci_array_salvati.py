#Script da lanciare nella cartella esterna contenente tutte le configurazioni

import numpy as np
import glob 

arr_list = []

for name in sorted(glob.glob('dati_configurazione*')): #sorted per ordinare, glob da solo non ordina
    #print(name)
    arr = np.load(name)
    arr_list.append(arr)
    data3 = np.concatenate(arr_list)
np.save('./configurazione_posizioni_completa', data3)

data3 = None
del data3

arr_list = []

for name in sorted(glob.glob('dati_msd*')): #sorted per ordinare, glob da solo non ordina
    #print(name)
    arr = np.load(name)
    arr_list.append(arr)
    data3 = np.concatenate(arr_list)
np.save('./msd_completo', data3)
    
    
#per unire degli array mettili tutti in una lista con append e poi concatena la lista con data = np.concatenate(lista). questi li unisce lungo l'asse 1
