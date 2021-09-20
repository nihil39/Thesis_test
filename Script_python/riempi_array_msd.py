import numpy as np
#import fileinput
import glob 
import os

data2 = np.empty(shape=(0,200,3))

# Carica msd

for name in sorted(glob.glob('*pulito*')): #sorted per ordinare, glob da solo non ordina
	#print(name)
	dati_tmp = np.loadtxt(name)
	#print("forma dei dati_tmp: ", dati_tmp.shape)
	#print("tipo dei dati_tmp: ", dati_tmp.dtype)
	data2 = np.concatenate((data2, dati_tmp.reshape((1,200,3))), axis=0)
	#print("forma dei dati: ", data.shape)
    
np.save('./dati_msd',data2)	
