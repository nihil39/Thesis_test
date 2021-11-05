import numpy as np

#array_msd_T044 = np.load('msd_completo_T044_P293.npy')
array_msd_T045 = np.load('../T045_P270_NT/msd_completo_T045_P270_NT.npy')

#array_msd_T047 = np.load('msd_completo_T047_P224.npy')

#array_msd_T050 = np.load('msd_completo_T050_P155.npy')
#array_msd_T052 = np.load('msd_completo_T052_P109_NT.npy')

#array_msd_T056 = np.load('msd_completo_T056_P017.npy')
#

#array_medie_T044 = np.zeros((200,3))
array_medie_T045 = np.zeros((200,3))
#array_medie_T047 = np.zeros((200,3))
#array_medie_T050 = np.zeros((200,3))
#array_medie_T052 = np.zeros((200,3))
#array_medie_T056 = np.zeros((200,3))
#
for j in range(200):
#    array_medie_T044[j, 0] = array_msd_T044[0, j, 0] # copia i tempi 
#    array_medie_T044[j, 1] = array_msd_T044[:, j, 1].mean()
#    array_medie_T044[j, 2] = array_msd_T044[:, j, 2].mean()
#  
     array_medie_T045[j, 0] = array_msd_T045[0, j, 0] # copia i tempi 
     array_medie_T045[j, 1] = array_msd_T045[:, j, 1].mean()
     array_medie_T045[j, 2] = array_msd_T045[:, j, 2].mean()

#    array_medie_T047[j, 0] = array_msd_T047[0, j, 0] # copia i tempi 
#    array_medie_T047[j, 1] = array_msd_T047[:, j, 1].mean() 
#    array_medie_T047[j, 2] = array_msd_T047[:, j, 2].mean()  
#    
#    array_medie_T050[j, 0] = array_msd_T050[0, j, 0]  # copia i tempi
#    array_medie_T050[j, 1] = array_msd_T050[:, j, 1].mean()
#    array_medie_T050[j, 2] = array_msd_T050[:, j, 2].mean()

#    array_medie_T052[j, 0] = array_msd_T052[0, j, 0]  # copia i tempi
#    array_medie_T052[j, 1] = array_msd_T052[:, j, 1].mean()
#    array_medie_T052[j, 2] = array_msd_T052[:, j, 2].mean()

#    
#    array_medie_T056[j, 0] = array_msd_T056[0, j, 0]  # copia i tempi
#    array_medie_T056[j, 1] = array_msd_T056[:, j, 1].mean()
#    array_medie_T056[j, 2] = array_msd_T056[:, j, 2].mean()
#  # array_medie[j, 0] = np.around(array_medie[j, 0], decimals = 3)
#
#np.save('msd_T044_P293_medie.npy', array_medie_T044)
np.save('msd_T045_P270_NT_medie.npy', array_medie_T045)
#np.save('msd_T047_P224_medie.npy', array_medie_T047)
#np.save('msd_T050_P155_medie.npy', array_medie_T050)
#np.save('msd_T052_P109_NT_medie.npy', array_medie_T052)
#np.save('msd_T056_P017_medie.npy', array_medie_T056)

