import h5py
import tensorflow as tf
from tensorflow import keras

#import matplotlib.pyplot as plt
#import numpy as np
#
model_name = 'DGCNN_k20_e30_bs16.h5'
#
#
#with h5py.File(filename, "r") as f:


configurazioni = np.load('configurazione_posizioni_completa.npy')
msd = np.load('msd_completo.npy')

configurazioni_train, configurazioni_test, msd_train, msd_test = train_test_split(configurazioni, msd, shuffle = True, train_size = 0.8, random_state = 1234) 

class Dataset(object): #Is object really useful?
    def __init__(self, partition='train', num_points = 4096): # va specificato il numero di punti anche alla riga 86?
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
    
    @propertyi #Label
    def y(self):
        return self._label

    def shuffle(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        shuffle_indices = np.arange(self.__len__())
        np.random.shuffle(shuffle_indices)
        for k in self._values:
            self._values[k] = self._values[k][shuffle_indices]
        self._label = self._label[shuffle_indices]  


#legge i dati di training e test
train = Dataset(partition = 'train', num_points = 4096)



model = keras.models.load_model(model_name)

#model.load_weights("model_checkpoints/DGCNN_modelbest.h5")


test_loss, test_mae = model.evaluate(test.X, test.y, verbose = 1)

predictions =  model.predict(test.X, verbose = 1) #prediction vs test.y, istogramma di test.y - prediction



#predict_=  model.predict(test.X) vero test.


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

