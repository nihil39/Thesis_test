#!/usr/bin/env python
# coding: utf-8

import os
import sys
import glob
import h5py
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime
#Importa configurazioni

#x(8000,4096,7) # tipo,x,y,z,vx,vy,vz
#y(8000,200,3)

#configurazioni = np.load('configurazione_posizioni_completa.npy')
#msd = np.load('msd_completo.npy')

conf_T044 = np.load('../../T044_P293/configurazione_iniziale_completa_T044_P293.npy')
conf_T045 = np.load('../../T045_P270_NT/configurazione_iniziale_completa_T045_P270_NT.npy')
conf_T047 = np.load('../../T047_P224/configurazione_iniziale_completa_T047_P224.npy')
conf_T049 = np.load('../../T049_P178_NT/configurazione_iniziale_completa_T049_P178_NT.npy')
conf_T050 = np.load('../../T050_P155/configurazione_iniziale_completa_T050_P155.npy')
conf_T052 = np.load('../../T052_P109_NT/configurazione_iniziale_completa_T052_P109_NT.npy')
conf_T056 = np.load('../../T056_P017/configurazione_iniziale_completa_T056_P017.npy')

msd_T044 = np.load('../../T044_P293/msd_completo_T044_P293.npy')
msd_T045 = np.load('../../T045_P270_NT/msd_completo_T045_P270_NT.npy')
msd_T047 = np.load('../../T047_P224/msd_completo_T047_P224.npy')
msd_T049 = np.load('../../T049_P178_NT/msd_completo_T049_P178_NT.npy')
msd_T050 = np.load('../../T050_P155/msd_completo_T050_P155.npy')
msd_T052 = np.load('../../T052_P109_NT/msd_completo_T052_P109_NT.npy')
msd_T056 = np.load('../../T056_P017/msd_completo_T056_P017.npy')

#Configurazione non training

#conf_iniz_tot = np.concatenate((conf_T044, conf_T047, conf_T050, conf_T056))
#msd_tot = np.concatenate((msd_T044, msd_T047, msd_T050, msd_T056))

#chiama 4 volte

conf_T044_train, conf_T044_test, msd_T044_train, msd_T044_test = train_test_split(conf_T044, msd_T044, shuffle = True, train_size = 0.8, random_state = 1234) 
conf_T045_train, conf_T045_test, msd_T045_train, msd_T045_test = train_test_split(conf_T045, msd_T045, shuffle = True, train_size = 0.8, random_state = 7894) 
conf_T047_train, conf_T047_test, msd_T047_train, msd_T047_test = train_test_split(conf_T047, msd_T047, shuffle = True, train_size = 0.8, random_state = 3478) 
conf_T049_train, conf_T049_test, msd_T049_train, msd_T049_test = train_test_split(conf_T049, msd_T049, shuffle = True, train_size = 0.8, random_state = 8978) 
conf_T050_train, conf_T050_test, msd_T050_train, msd_T050_test = train_test_split(conf_T050, msd_T050, shuffle = True, train_size = 0.8, random_state = 1912) 
conf_T052_train, conf_T052_test, msd_T052_train, msd_T052_test = train_test_split(conf_T052, msd_T052, shuffle = True, train_size = 0.8, random_state = 6782) 
conf_T056_train, conf_T056_test, msd_T056_train, msd_T056_test = train_test_split(conf_T056, msd_T056, shuffle = True, train_size = 0.8, random_state = 4532)

#configurazioni_train, configurazioni_test, msd_train, msd_test = train_test_split(configurazioni, msd, shuffle = True, train_size = 0.8, random_state = 1i234) 

# classe per leggere il dataset e formattarlo in modo appropriato per l'input della DGCNN 
# il dataset contiene tre tipi di oggetti:
# points: la point cloud che contiene le coordinate dei punti: shape (N, P, C_p) N=numero esempi, P=numero punti per ogni esempio, C_p=numero feature associate a ciascun punto
# features: le feature associate ad ogni punto (possono essere le stesse coordinate o queste + ulteriori features): shape (N, P, C_f)
# mask: una mask che ha valore 1 o 0 per mascherare punti non fisici (quando P di un dato evento è inferiore alla dimensione con cui si e' fissato P): shape (N,P,1) 

class Dataset(object): #Is object really useful?
    def __init__(self, partition = 'train', num_points = 4096): # va specificato il numero di punti anche alla riga 86?
        if partition == 'train':
           self.data, self.label = conf_T044_train, msd_T044_train
        elif partition == 'T044_test': 
           self.data, self.label = conf_T044_test, msd_T044_test # Test
        elif partition == 'T045_test': 
           self.data, self.label = conf_T045_test, msd_T045_test # Test
        elif partition == 'T047_test':
           self.data, self.label = conf_T047_test, msd_T047_test # Test
        elif partition == 'T049_test': 
           self.data, self.label = conf_T049_test, msd_T049_test # Test
        elif partition == 'T050_test':
           self.data, self.label = conf_T050_test, msd_T050_test # Test
        elif partition == 'T052_test': 
           self.data, self.label = conf_T052_test, msd_T052_test # Test
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
    
    @property #Label, what we have to predict from the features
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
test_T044  = Dataset(partition = 'T044_test',  num_points = 4096)
test_T045  = Dataset(partition = 'T045_test',  num_points = 4096)
test_T047  = Dataset(partition = 'T047_test',  num_points = 4096)
test_T049  = Dataset(partition = 'T049_test',  num_points = 4096)
test_T050  = Dataset(partition = 'T050_test',  num_points = 4096)
test_T052  = Dataset(partition = 'T052_test',  num_points = 4096)
test_T056  = Dataset(partition = 'T056_test',  num_points = 4096)


# mostra il contenuto dei dati
#print(train['points'].shape)
#print(test['points'].shape)
#print(train['features'].shape)
#print(test['features'].shape)
#print(train['mask'].shape)
#print(test['mask'].shape)


# DGCNN
# https://github.com/hqucms/ParticleNet/blob/master/tf-keras/tf_keras_model.py

# A shape is (N, P_A, C), B shape is (N, P_B, C)
# D shape is (N, P_A, P_B)
def batch_distance_matrix_general(A, B):
    with tf.name_scope('dmat'):
        r_A = tf.reduce_sum(A * A, axis=2, keepdims = True)
        r_B = tf.reduce_sum(B * B, axis=2, keepdims = True)
        m = tf.matmul(A, tf.transpose(B, perm = (0, 2, 1)))
        D = r_A - 2 * m + tf.transpose(r_B, perm = (0, 2, 1))
        return D
    
def knn(num_points, k, topk_indices, features):
    # topk_indices: (N, P, K)
    # features: (N, P, C)
    with tf.name_scope('knn'):
        queries_shape = tf.shape(features)
        batch_size = queries_shape[0]
        batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, num_points, k, 1))
        indices = tf.concat([batch_indices, tf.expand_dims(topk_indices, axis=3)], axis=3)  # (N, P, K, 2)
        return tf.gather_nd(features, indices)
    
def edge_conv(points, features, num_points, K, channels, with_bn=True, activation='relu', pooling='max', name='edgeconv'):
    """EdgeConv
    Args:
        K: int, number of neighbors
        in_channels: # of input channels
        channels: tuple of output channels
        pooling: pooling method ('max' or 'average')
    Inputs:
        points: (N, P, C_p)
        features: (N, P, C_0)
    Returns:
        transformed points: (N, P, C_out), C_out = channels[-1]
    """

    with tf.name_scope('edgeconv'):

        # distance
        D = batch_distance_matrix_general(points, points)  # (N, P, P)
        _, indices = tf.nn.top_k(-D, k=K + 1)  # (N, P, K+1)
        indices = indices[:, :, 1:]  # (N, P, K)

        fts = features
        knn_fts = knn(num_points, K, indices, fts)  # (N, P, K, C)
        knn_fts_center = tf.tile(tf.expand_dims(fts, axis=2), (1, 1, K, 1))  # (N, P, K, C)
        knn_fts = tf.concat([knn_fts_center, tf.subtract(knn_fts, knn_fts_center)], axis=-1)  # (N, P, K, 2*C)

        x = knn_fts
        for idx, channel in enumerate(channels):
            x = keras.layers.Conv2D(channel, kernel_size=(1, 1), strides=1, data_format='channels_last',
                                    use_bias=False if with_bn else True, kernel_initializer='glorot_normal', name='%s_conv%d' % (name, idx))(x)
            if with_bn:
                x = keras.layers.BatchNormalization(name='%s_bn%d' % (name, idx))(x)
            if activation:
                x = keras.layers.Activation(activation, name='%s_act%d' % (name, idx))(x)

        if pooling == 'max':
            fts = tf.reduce_max(x, axis=2)  # (N, P, C')
        else:
            fts = tf.reduce_mean(x, axis=2)  # (N, P, C')

        # shortcut
        sc = keras.layers.Conv2D(channels[-1], kernel_size=(1, 1), strides=1, data_format='channels_last',
                                 use_bias=False if with_bn else True, kernel_initializer='glorot_normal', name='%s_sc_conv' % name)(tf.expand_dims(features, axis=2))
        if with_bn:
            sc = keras.layers.BatchNormalization(name='%s_sc_bn' % name)(sc)
        sc = tf.squeeze(sc, axis=2)

        if activation:
            return keras.layers.Activation(activation, name='%s_sc_act' % name)(sc + fts)  # (N, P, C')
        else:
            return sc + fts


def _DGCNN_base(points, features=None, mask = None, setting = None, name = 'DGCNN_SG'):
    # points : (N, P, C_coord)
    # features:  (N, P, C_features), optional
    # mask: (N, P, 1), optional

    with tf.name_scope(name):
        if features is None:
            features = points
        
        if mask is not None:
            mask = tf.cast(tf.not_equal(mask, 0), dtype='float32')  # 1 if valid
            coord_shift = tf.multiply(999., tf.cast(tf.equal(mask, 0), dtype='float32'))  # make non-valid positions to 99   
            
        fts = tf.squeeze(keras.layers.BatchNormalization(name='%s_fts_bn' % name)(tf.expand_dims(features, axis=2)), axis=2)
        for layer_idx, layer_param in enumerate(setting.conv_params):
            K, channels = layer_param
            pts = tf.add(coord_shift, points) if layer_idx == 0 else tf.add(coord_shift, fts)
            fts = edge_conv(pts, fts, setting.num_points, K, channels, with_bn=True, activation='relu',
                            pooling=setting.conv_pooling, name='%s_%s%d' % (name, 'EdgeConv', layer_idx))

        if mask is not None:
            fts = tf.multiply(fts, mask)

        pool = tf.reduce_mean(fts, axis=1)  # (N, C)

        if setting.fc_params is not None:
            x = pool
            for layer_idx, layer_param in enumerate(setting.fc_params):
                units, drop_rate = layer_param
                x = keras.layers.Dense(units, activation='relu')(x)
                if drop_rate is not None and drop_rate > 0:
                    x = keras.layers.Dropout(drop_rate)(x)


            #out = keras.layers.Dense(setting.num_class, activation='softmax')(x)
            out = keras.layers.Dense(setting.num_class)(x)
            return out  # (N, num_classes)
        else:
            return pool


class _DotDict:
    pass

def get_DGCNN(num_classes, input_shapes):
    """
    Parameters
    ----------
    num_classes : int
        Number of output classes.
    input_shapes : dict
        The shapes of each input (`points`, `features`, `mask`).
    """
    setting = _DotDict()
    setting.num_class = num_classes
    # conv_params: list of tuple in the format (K, (C1, C2, C3)) (k primi vicini, numero di filtri C1 C2 C3)
    setting.conv_params = [
        (30, (64, 64, 64)), 
        (30, (64, 64, 64)),
        (30, (128, 128, 128)),
        (30, (256, 256, 256)),
        ]
    # conv_pooling: 'average' or 'max'
    setting.conv_pooling = 'average' #prova a modificare questo
    # fc_params: list of tuples in the format (C, drop_rate)
    setting.fc_params = [
        (512, 0.5),
        (256, 0.5),
        ]
    setting.num_points = input_shapes['points'][0]

    points = keras.Input(name='points', shape=input_shapes['points'])
    features = keras.Input(name='features', shape=input_shapes['features']) if 'features' in input_shapes else None
    mask = keras.Input(name='mask', shape=input_shapes['mask']) if 'mask' in input_shapes else None
    outputs = _DGCNN_base(points, features, mask, setting, name='DGCNN_SG')

    return keras.Model(inputs=[points, features, mask], outputs=outputs, name='DGCNN_SG')

num_classes = 2 #number of points to be predicted
input_shapes = {k:train[k].shape[1:] for k in train.X}
print(input_shapes)
model = get_DGCNN(num_classes, input_shapes)


import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

def lr_schedule(epoch): # prova 3e-4 
    lr = 1e-3
    if epoch > 10:
        lr *= 0.1
    logging.info('Learning rate: %f'%lr)
    return lr

model.compile(loss = 'mse',
              optimizer = keras.optimizers.Adam(learning_rate=lr_schedule(0)),
              metrics = ['mae'])
model.summary()


# Prepare model saving directory.
import os
save_dir = 'model_checkpoints'

#model_name = 'DGCNN_modelbest.h5'
#model_name = 'DGCNN_k10_e30_bs32_cpMax-epoch{epoch:02d}-mae{val_mae:.2f}.h5'

model_name = 'DGCNN_T044_Training_2.h5'

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = keras.callbacks.ModelCheckpoint(filepath = filepath,
                             monitor = 'val_mae', # cosa sceglie come best
                             verbose = 1,
                             save_best_only = True)

lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)
progress_bar = keras.callbacks.ProgbarLogger()
tensorboard_log_folder = "./tensorboard_logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir = tensorboard_log_folder,
                                                    update_freq = 'epoch',
                                                    write_graph = True,
                                                    write_images = True)

callbacks = [checkpoint, lr_scheduler, progress_bar,tensorboard_callback]

# TRAINING PARAMETERS

batch_size = 8 #era 32
epochs = 30 # 30 valore iniziale

# model.fit() to train the model https://www.tensorflow.org/api_docs/python/tf/keras/Model
train.shuffle()
history = model.fit(train.X, train.y,
          batch_size = batch_size,
          epochs = epochs,
          #validation_data = (test.X, test.y),
          validation_split = 0.2,
          shuffle = True,
          callbacks = callbacks)

#model.load_weights("model_checkpoints/DGCNN_modelbest.h5")

model.load_weights("model_checkpoints/DGCNN_T044_Training_2.h5")
#model.load_weights("model_checkpoints/prova.h5")

#test_loss, test_mae = model.evaluate(test.X, test.y, verbose = 1) #Restituisce prima la loss e poi le metrics definite in model.compile

#predictions =  model.predict(test.X, verbose = 1) # in output qualcosa con la forma di test.y
predictions_T044 = model.predict(test_T044.X, verbose = 1) #predicted value for new unlabeled data (or pretend that you do not have a label)
predictions_T045 = model.predict(test_T045.X, verbose = 1) #predicted value for new unlabeled data (or pretend that you do not have a label)
predictions_T047 = model.predict(test_T047.X, verbose = 1) #predicted value for new unlabeled data (or pretend that you do not have a label)
predictions_T049 = model.predict(test_T049.X, verbose = 1) #predicted value for new unlabeled data (or pretend that you do not have a label)
predictions_T050 = model.predict(test_T050.X, verbose = 1) #predicted value for new unlabeled data (or pretend that you do not have a label)
predictions_T052 = model.predict(test_T052.X, verbose = 1) #predicted value for new unlabeled data (or pretend that you do not have a label)
predictions_T056 = model.predict(test_T056.X, verbose = 1) #predicted value for new unlabeled data (or pretend that you do not have a label)
#
##np.save('test_loss_k20_e30_bs16_cpMax.npy', test_loss)
##np.save('test_mae_k20_e30_bs16_cpMax.npy', test_mae)
#
np.save('predictions_T044_Training_2_T044_test.npy', predictions_T044)
np.save('test_y_T044_Training_2_T044_test.npy', test_T044.y)

np.save('predictions_T044_Training_2_T045_test.npy', predictions_T045)
np.save('test_y_T044_Training_2_T045_test.npy', test_T045.y)
#
np.save('predictions_T044_Training_2_T047_test.npy', predictions_T047)
np.save('test_y_T044_Training_2_T047_test.npy', test_T047.y)

np.save('predictions_T044_Training_2_T049_test.npy', predictions_T049)
np.save('test_y_T044_Training_2_T049_test.npy', test_T049.y)
#
np.save('predictions_T044_Training_2_T050_test.npy', predictions_T050)
np.save('test_y_T044_Training_2_T050_test.npy', test_T050.y)

np.save('predictions_T044_Training_2_T052_test.npy', predictions_T052)
np.save('test_y_T044_Training_2_T052_test.npy', test_T052.y)
#
np.save('predictions_T044_Training_2_T056_test.npy', predictions_T056)
np.save('test_y_cpMax_T044_Training_2_T056_test.npy', test_T056.y)

### PLOT ###
# Salvare history, mae, val mae, loss, val loss in un altro array
mae = history.history['mae']
val_mae = history.history['val_mae']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(loss) + 1)

plt.subplot(1, 2, 1)
plt.plot(epochs_range, loss, label = 'Training Loss')
plt.plot(epochs_range, val_loss, label = 'Validation Loss')
plt.legend(loc = 'upper right')
plt.title('Training and Validation Loss')

#plt.show()
#plt.figure(figsize = (8, 8))

plt.subplot(1, 2, 2)
plt.plot(epochs_range, mae, label = 'Training MAE')
plt.plot(epochs_range, val_mae, label = 'Validation MAE')
plt.legend(loc = 'upper right')
plt.title('Training and Validation MAE')

plt.savefig('TaV_Loss_T044_Training_2.pdf')

###################

#from sklearn.metrics import classification_report, confusion_matrix
#
#pred = model.predict(test.X)
#matrix = confusion_matrix(test.y.argmax(axis=1), pred.argmax(axis=1), normalize='true')
#
#
##confusion matrix
#def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
#    """pretty print for confusion matrixes"""
#    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
#    empty_cell = " " * columnwidth
#    # Print header
#    print("    " + empty_cell, end=" ")
#    for label in labels:
#        print("%{0}s".format(columnwidth) % label, end=" ")
#    print()
#
#    # Print rows
#    for i, label1 in enumerate(labels):
#        print("    %{0}s".format(columnwidth) % label1, end=" ")
#        for j in range(len(labels)):
#            cell = "%{0}.2f".format(columnwidth) % cm[i, j]
#            if hide_zeroes:
#                cell = cell if float(cm[i, j]) != 0 else empty_cell
#            if hide_diagonal:
#                cell = cell if i != j else empty_cell
#            if hide_threshold:
#                cell = cell if cm[i, j] > hide_threshold else empty_cell
#            print(cell, end=" ")
#        print()
#
#
#le_label=[ str(i) for i in range(test.y.shape[1])]
#print_cm(matrix,le_label)


