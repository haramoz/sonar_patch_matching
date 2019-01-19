import numpy as np
import random
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
import csv

import keras.backend as K
from keras.optimizers import RMSprop,Adam
from keras.utils import np_utils
from keras.models import Model,Sequential
from keras.models import model_from_json
import keras.initializers
from keras.losses import binary_crossentropy
#from keras.callbacks import EarlyStopping,ModelCheckpoint

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import pickle

seed = 7
np.random.seed(seed)

def process_data():
    f = h5py.File('/home/amalli2s/thesis/keras/matchedImagesSplitClasses-2017-02-24-17-39-44-96-96-split-val0.15-tr0.7-tst0.15.hdf5','r')
    X_train = f['X_train'].value
    y_train = f['y_train'].value
    X_test = f['X_val'].value
    y_test = f['y_val'].value
    #X_train_, X_val, y_train_, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)
    
    return X_train,y_train,X_test,y_test

def make_mcdropout_predictor(model): 

    f = K.function([model.layers[0].input, K.learning_phase()],[model.layers[-1].output])
    return lambda x: f([x, 1])[0]

X_train,y_train,X_test,y_test = process_data() 

# load json and create model
#file_name = 'keras_densenet_twochannel_7Nov_14400.9593'
#file_name = 'keras_densenet_twochannel_14Dec_00000.9664'
#file_name = 'keras_densenet_twochannel_9Nov_14400.9725'
#file_name = 'keras_densenet_twochannel_14Dec_00000.9632'
file_name = 'keras_densenet_twochannel_14Dec_00000.9704'

json_file = open(file_name+'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(file_name+".h5")
print("Loaded model from disk")
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#MC dropout analysis
mc_model = make_mcdropout_predictor(loaded_model)
NUM_MCDROPOUT_ITERATIONS = 20
samples = []
a = 1000
b = 1020
pred = loaded_model.predict(X_test)
#print("pred: \n",pred[a:b])

for i in range(NUM_MCDROPOUT_ITERATIONS):
    samples.append(mc_model(X_test))

samples = np.array(samples)
mean_pred = np.mean(samples, axis=0)
ground_truth = y_test
prediction= pred
std_pred = np.std(samples, axis=0)

print(len(mean_pred))
print(type(mean_pred))

"""with open('mc_dropout_dtc_analysis.csv', mode='w') as mc_file:
    mc_writer = csv.writer(mc_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    for i in range(len(mean_pred)):
        #print(np.round(prediction[i],3)[0],np.round(mean_pred[i],3)[0],np.round(std_pred[i],3)[0],ground_truth[i])
        mc_writer.writerow([np.round(prediction[i],3)[0],np.round(mean_pred[i],3)[0],np.round(std_pred[i],3)[0],ground_truth[i]])
    mc_file.close()
"""
