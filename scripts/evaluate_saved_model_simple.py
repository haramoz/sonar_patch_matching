import numpy as np
import random
import h5py
import tensorflow as tf
from keras.models import load_model
from keras.models import model_from_json
from keras.losses import binary_crossentropy
from keras.optimizers import RMSprop,Adam

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score


def process_data():
    f = h5py.File('matchedImagesSplitClasses-2017-02-24-17-39-44-96-96-split-val0.15-tr0.7-tst0.15.hdf5','r')
    X_train = f['X_train'].value
    y_train = f['y_train'].value
    X_val = f['X_val'].value
    y_val = f['y_val'].value

    return X_train,y_train,X_val,y_val


X_train,y_train,X_val,y_val = process_data()
file_name = 'model'
# load json and create model
json_file = open(file_name+'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(file_name+".h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(X_val,y_val, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

pred = model.predict(X_val)
auc_score = roc_auc_score(y_val,pred)
print("auc_score ------------------> ",auc_score)
