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
from keras.optimizers import RMSprop,Adam,Adadelta
from keras.utils import np_utils
from keras.models import Model,Sequential
from keras.models import model_from_json
from keras.models import load_model
import keras.initializers
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from keras.utils import multi_gpu_model

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

from keras_contrib.applications import DenseNet
#import keras_densenet

import argparse

seed = 7
np.random.seed(seed)

def process_data():
    f = h5py.File('/home/amalli2s/thesis/keras/matchedImagesSplitClasses-2017-02-24-17-39-44-96-96-split-val0.15-tr0.7-tst0.15.hdf5','r')
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

def fit_model(X_train,y_train,X_val,y_val,G):
    epochs = 5
    es_patience = 5
    lr_patience = 3
    dropout1 = None
    depth = 25 #40
    nb_dense_block = 3
    nb_filter = 18
    growth_rate = 18
    lr = 3E-1
    weight_file = 'keras_densenet_simple_wt_30Sept.h5'
    bn = True 
    reduction_ = 0.5    
    nb_classes = 1
    img_dim = (2,96,96) 
    n_channels = 2 

    
    model  = DenseNet(depth=depth, nb_dense_block=nb_dense_block,
                 growth_rate=growth_rate, nb_filter=nb_filter,
                 dropout_rate=dropout1,activation='sigmoid',
                 input_shape=img_dim,include_top=True,
                 bottleneck=bn,reduction=reduction_,
                 classes=nb_classes,pooling='avg',
                 weights=None)
    

    model.summary()
    opt = Adam(lr=lr)
    parallel_model = multi_gpu_model(model, gpus=G)
    
    parallel_model.compile(loss=binary_crossentropy, optimizer=Adadelta(), metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', patience=es_patience,verbose=1)
    #es = EarlyStopping(monitor='val_acc', patience=es_patience,verbose=1,restore_best_weights=True)
    checkpointer = ModelCheckpoint(filepath=weight_file,verbose=1, save_best_only=True)

    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=lr_patience, min_lr=0.5e-6,verbose=1)

    parallel_model.fit(X_train,y_train,
          batch_size=64*G,
          epochs=epochs,
          callbacks=[es,lr_reducer,checkpointer],
          validation_data=(X_val,y_val),
          verbose=2)
    
    score, acc = parallel_model.evaluate(X_val, y_val)
    print('current Test accuracy:', acc)
    pred = parallel_model.predict(X_val)
    auc_score = roc_auc_score(y_val,pred)
    print("current auc_score ------------------> ",auc_score)

    """model = load_model(weight_file) #This is the best model
    score, acc = model.evaluate(X_val, y_val)
    print('Best saved model Test accuracy:', acc)
    pred = model.predict(X_val)
    auc_score = roc_auc_score(y_val,pred)
    print("best saved model auc_score ------------------> ",auc_score)"""


    return auc_score,parallel_model

if __name__ == '__main__':
    
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-g", "--gpus", type=int, default=2,
	help="# of GPUs to use for training")
    args = vars(ap.parse_args())
    G = args["gpus"]
    print("Gpus in the use",G)
    X_train,y_train,X_val,y_val = process_data()
    scores = []
    file_name = 'model-1-10-3pm'
    for i in range(3):
        score,model = fit_model(X_train,y_train,X_val,y_val,G)
     
        score = np.round(score,3)
        scores.append(score)
        if(score > .93):
            model_json = model.to_json()
            with open(file_name+".json", "w") as json_file:
                json_file.write(model_json)
            model.save_weights(file_name+".h5")
            print("Saved model to disk")
            del model

    print(scores)
    mean = np.round(np.mean(scores),3)
    std = np.round(np.std(scores),3)
    print(mean,u'\u00B1',std)
