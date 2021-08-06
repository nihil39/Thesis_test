    import numpy as np
    ​
    data = np.empty(shape=(0,4096,7))
    target = np.empty(shape=(0,200)))
    for i in range(num_eventi):
      dat_tmp = np.empty(shape=(0,7))
      for k in range(4096):
        x = ..
        y = ..
    ​
        vx = ..
        ..
        tipo = ..
    ​
        tmp = np.array([x,y,z,vx,vy,vz,tipo])
        dat_tmp = np.concatenate((dat_tmp, tmp.reshape((1,7))), axis=0)
    ​
      data = np.concatenate((data, dat_tmp.reshape((1,4096,7))), axis=0)
    ​
      msq_values = np.zeros(shape=(200))
    ​
      for j in range(200):
        msq_values[j] = ...
    ​
      target = np.concatenate((target, msq_values.reshape((1,200))), axis=0)
       
    print(data.shape)
    ​
    np.savez('nome_file', data, target, data=data, target=target)
    
    ####################
    
    import numpy as np

#pippo = np.ones(shape=(4096,7))

#np.savetxt('pincopallino.dat',pippo)
dati123 = np.loadtxt('pincopallino.dat')
print(dati123.shape)
