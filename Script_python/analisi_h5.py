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

model = keras.models.load_model(model_name)

#model.load_weights("model_checkpoints/DGCNN_modelbest.h5")


test_loss, test_mae = model.evaluate(test.X, test.y, verbose = 2)



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

