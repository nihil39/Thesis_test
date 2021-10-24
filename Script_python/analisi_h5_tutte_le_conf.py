import h5py
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split

#Comment the line below to use the GPU
#tf.config.set_visible_devices([], 'GPU')

#import matplotlib.pyplot as plt
#import numpy as np
#
model_name = 'DGCNN_k20_e30_bs16_cpMax.h5'
#
#
#with h5py.File(filename, "r") as f:


conf_T044 = np.load('../../T044_P293/configurazione_iniziale_completa_T044_P293.npy')
conf_T047 = np.load('../../T047_P224/configurazione_iniziale_completa_T047_P224.npy')
conf_T050 = np.load('../../T050_P155/configurazione_iniziale_completa_T050_P155.npy')
conf_T056 = np.load('../../T056_P017/configurazione_iniziale_completa_T056_P017.npy')

msd_T044 = np.load('../../T044_P293/msd_completo_T044_P293.npy')
msd_T047 = np.load('../../T047_P224/msd_completo_T047_P224.npy')
msd_T050 = np.load('../../T050_P155/msd_completo_T050_P155.npy')
msd_T056 = np.load('../../T056_P017/msd_completo_T056_P017.npy')

conf_T044_train, conf_T044_test, msd_T044_train, msd_T044_test = train_test_split(conf_T044, msd_T044, shuffle = True, train_size = 0.8, random_state = 1234) 
conf_T047_train, conf_T047_test, msd_T047_train, msd_T047_test = train_test_split(conf_T047, msd_T047, shuffle = True, train_size = 0.8, random_state = 5678) 
conf_T050_train, conf_T050_test, msd_T050_train, msd_T050_test = train_test_split(conf_T050, msd_T050, shuffle = True, train_size = 0.8, random_state = 9012) 
conf_T056_train, conf_T056_test, msd_T056_train, msd_T056_test = train_test_split(conf_T056, msd_T056, shuffle = True, train_size = 0.8, random_state = 3456)

conf_tot_train = np.concatenate((conf_T044_train, conf_T047_train, conf_T050_train, conf_T056_train))
msd_tot_train = np.concatenate((msd_T044_train, msd_T047_train, msd_T050_train, msd_T056_train))

class Dataset(object): #Is object really useful?
    def __init__(self, partition='train', num_points = 4096):  
        if partition == 'train':
           self.data, self.label = conf_tot_train, msd_tot_train
        elif partition == 'T044_test': 
           self.data, self.label = conf_T044_test, msd_T044_test # Test
        elif partition == 'T047_test':
           self.data, self.label = conf_T047_test, msd_T047_test # Test
        elif partition == 'T050_test':
           self.data, self.label = conf_T050_test, msd_T050_test # Test
        elif partition == 'T056_test':
           self.data, self.label = conf_T056_test, msd_T056_test # Test
        
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
        self._label = self.label[:,[100,199],1]  #msd[tutte le 8000 configurazioni, tre valori, msd particelle tipo A]
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
#test = Dataset(partition = 'test', num_points = 4096)

test_T044 = Dataset(partition = 'T044_test', num_points = 4096)
test_T047 = Dataset(partition = 'T047_test', num_points = 4096)
test_T050 = Dataset(partition = 'T050_test', num_points = 4096)
test_T056 = Dataset(partition = 'T056_test', num_points = 4096)

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


#test_loss_T044, test_mae_T044 = model.evaluate(test_T044.X, test_T044.y, verbose = 1)
#test_loss_T047, test_mae_T047 = model.evaluate(test_T047.X, test_T047.y, verbose = 1)
test_loss_T050, test_mae_T050 = model.evaluate(test_T050.X, test_T050.y, verbose = 1)
test_loss_T056, test_mae_T056 = model.evaluate(test_T056.X, test_T056.y, verbose = 1)

#np.save('test_loss_T044.npy', test_loss_T044)
#np.save('test_mae_T044.npy', test_mae_T044)

#np.save('test_loss_T047.npy', test_loss_T047)
#np.save('test_mae_T047.npy', test_mae_T047)

np.save('test_loss_T050.npy', test_loss_T050)
np.save('test_mae_T050.npy', test_mae_T050)

np.save('test_loss_T056.npy', test_loss_T056)
np.save('test_mae_T056.npy', test_mae_T056)

#predictions_T044 = model.predict(test_T044.X, verbose = 1) #prediction vs test.y, istogramma di test.y - prediction, predicted value for new unlabeled data (or pretend that you do not have a label)
#
#predictions_T047 = model.predict(test_T047.X, verbose = 1) #prediction vs test.y, istogramma di test.y - prediction, predicted value for new unlabeled data (or pretend that you do not have a label)
#
#predictions_T050 = model.predict(test_T050.X, verbose = 1) #prediction vs test.y, istogramma di test.y - prediction, predicted value for new unlabeled data (or pretend that you do not have a label)
#
#predictions_T056 = model.predict(test_T056.X, verbose = 1) #prediction vs test.y, istogramma di test.y - prediction, predicted value for new unlabeled data (or pretend that you do not have a label)
#
##np.save('test_loss_k20_e30_bs16_cpMax.npy', test_loss)
##np.save('test_mae_k20_e30_bs16_cpMax.npy', test_mae)
#
#np.save('predictions_k20_e30_bs16_cpMax_T044.npy', predictions_T044)
#np.save('test_y_k20_e30_bs_16_cpMax_T044.npy', test_T044.y)
#
#np.save('predictions_k20_e30_bs16_cpMax_T047.npy', predictions_T047)
#np.save('test_y_k20_e30_bs_16_cpMax_T047.npy', test_T047.y)
#
#np.save('predictions_k20_e30_bs16_cpMax_T050.npy', predictions_T050)
#np.save('test_y_k20_e30_bs_16_cpMax_T050.npy', test_T050.y)
#
#np.save('predictions_k20_e30_bs16_cpMax_T056.npy', predictions_T056)
#np.save('test_y_k20_e30_bs_16_cpMax_T056.npy', test_T056.y)


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

