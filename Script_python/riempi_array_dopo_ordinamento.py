#Programma lanciato da carica_dati_in_array_numpy. Questo programma carica in memoria e salva in degli arry numpy i dati delle posizioni e del msd della singola cartella 


import numpy as np
#import fileinput
import glob 
#import os

#Salva le configurazioni di UNA cartella nel file dati_configurazione

data = np.empty(shape=(0,4096,7))
data2 = np.empty(shape=(0,4096,7))

for name in sorted(glob.glob('*ordinato*')): #sorted per ordinare, glob da solo non ordina
	#print(name)
	dati_tmp = np.loadtxt(name)
	#print("forma dei dati_tmp: ", dati_tmp.shape)
	#print("tipo dei dati_tmp: ", dati_tmp.dtype)
	data = np.concatenate((data, dati_tmp.reshape((1,4096,7))), axis=0)
	#print("forma dei dati: ", data.shape)
	
np.save('./dati_configurazione',data)	

data = None
del data

for name in sorted(glob.glob('*pulito*')): #sorted per ordinare, glob da solo non ordina
	print(name)
	dati_tmp = np.loadtxt(name)
	print("forma dei dati_tmp: ", dati_tmp.shape)
	print("tipo dei dati_tmp: ", dati_tmp.dtype)
	data2 = np.concatenate((data, dati_tmp.reshape((1,4096,7))), axis=0)
	#print("forma dei dati: ", data.shape)
	
np.save('./dati_msd',data2)	

#	dati_config = 
#	
#	
#	print("forma dei dati: ", dati_configurazione.shape)
#	print("tipo dei dati: ", dati_configurazione.dtype)
#	dat_tmp = dati_configurazione
	
#array = np.loadtxt(fileinput.input(glob.glob('*ordinato*'))) #https://gist.github.com/salticus/a462912dfff90c9bded954c48f916f64
