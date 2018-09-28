import numpy as np
import random
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt

import keras.backend as K
from keras.initializers import RandomNormal
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Input,concatenate,Concatenate
from keras.layers import normalization, BatchNormalization, Lambda
from keras.layers import Flatten, Conv2D,MaxPooling2D
from keras.regularizers import l2
from keras.optimizers import RMSprop,Adam
from keras.utils import np_utils
from keras.models import Model,Sequential
from keras.models import model_from_json
import keras.initializers
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

from keras_contrib.applications import DenseNet
#import keras_densenet

import pickle

seed = 7
np.random.seed(seed)

def process_data():
    f = h5py.File('matchedImagesSplitClasses-2017-02-24-17-39-44-96-96-split-val0.15-tr0.7-tst0.15.hdf5','r')
    ln_training = 39840
    X_train = f['X_train'].value
    X_train_resize = X_train[0:ln_training,:,:,:]
    #print(X_train_resize.shape)
    y_train = f['y_train'].value
    y_train_resize = y_train[0:ln_training,]
    y_train_categorical = np_utils.to_categorical(y_train_resize, 2)
    #X_reshaped = X_train_resize.reshape(*X_train_resize.shape[:1], -2)
    #print(X_reshaped.shape)
    ln_validation = 7440
    X_val = f['X_val'].value
    X_val_resize = X_val[0:ln_validation,:,:,:]

    #X_val_reshaped = X_val
    #X_val_reshaped = X_val_resize.reshape(*X_val_resize.shape[:1], -2)

    y_val = f['y_val'].value
    y_val_reshaped = y_val[0:ln_validation]
    y_val_categorical = np_utils.to_categorical(y_val_reshaped, 2)
    return X_train,y_train,X_val,y_val

def fit_model(X_train,y_train,X_val,y_val):
    epochs = 20
    #input_shape = (1,96,96)
    patience_ = 5
    #dense_filter = 512
    #dropout = 0.76
    dropout1 = None
    depth = 13 #40
    nb_dense_block = 3
    nb_filter = 18
    growth_rate = 12
    weight_decay = 1E-4
    lr = 3E-5
    
    nb_classes = 1
    img_dim = (2,96,96) 
    n_channels = 2 

    
    model  = DenseNet(depth=depth, nb_dense_block=nb_dense_block,
                 growth_rate=growth_rate, nb_filter=nb_filter,
                 dropout_rate=dropout1,activation='sigmoid',
                 input_shape=img_dim,include_top=True,
                 bottleneck=True,reduction=0.5,
                 classes=nb_classes,pooling='avg',
                 weights=None)
    

    model.summary()
    opt = Adam(lr=lr)
    model.compile(loss=binary_crossentropy, optimizer=opt, metrics=['accuracy'])

    es = EarlyStopping(monitor='val_acc', patience=patience_,verbose=1)
    #es = EarlyStopping(monitor='val_acc', patience=patience_,verbose=1,restore_best_weights=True)
    checkpointer = ModelCheckpoint(filepath='keras_densenet_simple_weights.hdf5',
                                   verbose=1,
                                   save_best_only=True)

    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),
                               cooldown=0, patience=3, min_lr=0.5e-6,verbose=1)
    model.fit(X_train,y_train,
          batch_size=64,
          epochs=epochs,
          callbacks=[es,lr_reducer],
          validation_data=(X_val,y_val),
          verbose=2)

    score, acc = model.evaluate(X_val, y_val)
    print('Test accuracy:', acc)
    pred = model.predict(X_val)

    threshold = 0.6
    pred_scores2 = (pred>threshold).astype(int) 

    test_acc2 = accuracy_score(y_val,pred_scores2)
                                
    auc_score = roc_auc_score(y_val,pred)

    print('Test accuracy 0.6:', test_acc2)
    print("auc_score ------------------> ",auc_score)

    return auc_score,model   

if __name__ == '__main__':
    
    X_train,y_train,X_val,y_val = process_data()
    scores = []
    for i in range(10):
        score,model = fit_model(X_train,y_train,X_val,y_val)
     
        score = np.round(score,3)
        scores.append(score)
        if(score > .93):
            model_json = model.to_json()
            with open("model-27-9-6pm.json", "w") as json_file:
                json_file.write(model_json)
            model.save_weights("model-27-9-6pm.h5")
            print("Saved model to disk")
            del model

    print(scores)
    mean = np.round(np.mean(scores),3)
    std = np.round(np.std(scores),3)
    print(mean,u'\u00B1',std)
