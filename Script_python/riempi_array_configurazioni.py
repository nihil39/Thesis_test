import numpy as np
#import fileinput
import glob 
import os

data = np.empty(shape=(0,4096,7))

for name in sorted(glob.glob('*ordinato*')): #sorted per ordinare, glob da solo non ordina
	#print(name)
	dati_tmp = np.loadtxt(name)
	#print("forma dei dati_tmp: ", dati_tmp.shape)
	#print("tipo dei dati_tmp: ", dati_tmp.dtype)
	data = np.concatenate((data, dati_tmp.reshape((1,4096,7))), axis=0)
	#print("forma dei dati: ", data.shape)
	
np.save('./dati_configurazione',data)	

#data = None
