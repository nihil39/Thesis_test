import h5py
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split

#Comment the line below to use the GPU
tf.config.set_visible_devices([], 'GPU')

#import matplotlib.pyplot as plt
#import numpy as np
#
model_name = 'DGCNN_k10_e30_bs32_cpMax.h5'
#
#
#with h5py.File(filename, "r") as f:


configurazioni = np.load('configurazione_posizioni_completa.npy')
msd = np.load('msd_completo.npy')

configurazioni_train, configurazioni_test, msd_train, msd_test = train_test_split(configurazioni, msd, shuffle = True, train_size = 0.8, random_state = 1234) 

class Dataset(object): #Is object really useful?
    def __init__(self, partition='train', num_points = 4096):  
        if partition == 'train':
           self.data, self.label = configurazioni_train, msd_train
        else:
           self.data, self.label = configurazioni_test, msd_test
        self.num_points = num_points
        self._values = {}
        self._label = None
        self._load()
        
    def __len__(self):
        return len(self._label)

## Se servisse randomizzare le 4096 configurazioni iniziali delle particelle ##
#    lista = np.arange(4096)
#np.random.shuffle(lista)
#
#print(lista)
#lista[:1024]:w
#pointcloud = self.data[:, lista[:1024], :3]
    def _load(self):
        pointcloud = self.data[:, :4096, 1:4] # Configurazioni [configurazioni, numero particelle, tipo- coordinate- velocita']
        mask = np.ones(shape=(pointcloud.shape[0],pointcloud.shape[1],1))
        features = self.data[:, :4096, :] 
        self._label = self.label[:,[0,100,199],1]  #msd[tutte le 8000 configurazioni, tre valori, msd particelle tipo A]
        self._values['points'] = pointcloud
        self._values['features'] = features
        self._values['mask'] = mask
    
    def __getitem__(self, key):
        if key=='label':
            return self._label
        else:
            return self._values[key]
    
    @property #Features and validation ? points features and mask
    def X(self):
        return self._values
    
    @property #Label, known value for old data
    def y(self):
        return self._label

    def shuffle(self, seed = None):
        if seed is not None:
            np.random.seed(seed)
        shuffle_indices = np.arange(self.__len__())
        np.random.shuffle(shuffle_indices)
        for k in self._values:
            self._values[k] = self._values[k][shuffle_indices]
        self._label = self._label[shuffle_indices]  


#legge i dati di training e test
train = Dataset(partition = 'train', num_points = 4096)
test = Dataset(partition = 'test', num_points = 4096)

import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

def lr_schedule(epoch): # prova 3e-4 
    lr = 1e-3
    if epoch > 10:
        lr *= 0.1
    logging.info('Learning rate: %f'%lr)
    return lr

model = keras.models.load_model(model_name)
#model.load_weights("model_checkpoints/DGCNN_modelbest.h5")

model.compile(loss = 'mse',
              optimizer = keras.optimizers.Adam(learning_rate = lr_schedule(0)),
              metrics = ['mae'])


test_loss, test_mae = model.evaluate(test.X, test.y, verbose = 1)

predictions = model.predict(test.X, verbose = 1) #prediction vs test.y, istogramma di test.y - prediction, predicted value for new unlabeled data (or pretend that you do not have a label)


np.save('test_loss_k10_e30_bs32_cpMax.npy', test_loss)
np.save('test_mae_k10_e30_bs32_cpMax.npy', test_mae)
np.save('predictions_k10_e30_bs32_cpMax.npy', predictions)
np.save('test_y_k10_e30_bs32_cpMax.npy', test.y)


# perche' 1600,3 ? valore del msd nel punto 199 troppo basso

#def read_hdf5(filename):
#
#    weights = {}
#    
#    keys = []
#    with h5py.File(filename, 'r') as f: # open file
#        f.visit(keys.append) # append all keys to list
#        for key in keys:
#            if ':' in key: # contains data if ':' in key
#                print(f[key].name)
#                weights[f[key].name] = f[key].value
#    return weights
#
#if __name__ == '__main__':
#    filename = 'DGCNN_k20_e30_bs16.h5'
#    
#    weights = read_hdf5(filename)

