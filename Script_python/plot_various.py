#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
#from statistics import mean 

#a = np.array([1,100,4,5,-8,6,7,90000])

#print(np.average(a))


#with open('dati_KALJ_msd_log', 'r') as f:
with open('dati.txt', 'r') as f:
    lines = f.readlines()
    x = [float(line.split()[0]) for line in lines]
    y = [float(line.split()[1]) for line in lines]
    z = [float(line.split()[2]) for line in lines]
    #k = 2*np.exp(np.log(x)) + 2
    


fig, ax = plt.subplots()
    
plt.xlabel('intervalli di tempo')
plt.ylabel('<r²>')
plt.xscale("log")
plt.yscale("log")
ax.plot(x, y, '-b', label = 'part A')
ax.plot(x, z, '-r', label = 'part B')
#ax.plot(x, k, '-y', label = 'f')

leg = ax.legend();

plt.show()
