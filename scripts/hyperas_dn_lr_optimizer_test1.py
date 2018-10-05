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
from keras.optimizers import RMSprop,Adam,Adadelta,Adamax,Nadam
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
from sklearn.model_selection import train_test_split

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

from keras_contrib.applications import DenseNet
#import keras_densenet

import argparse

def process_data():
    random_seed = 7

    f = h5py.File('/home/amalli2s/thesis/keras/matchedImagesSplitClasses-2017-02-24-17-39-44-96-96-split-val0.15-tr0.7-tst0.15.hdf5','r')
    X_train = f['X_train'].value
    y_train = f['y_train'].value
    X_test = f['X_val'].value
    y_test = f['y_val'].value
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_seed)
 
    return X_train,y_train,X_val,y_val,X_test,y_test

def data():
    X_train,y_train,X_val,y_val,X_test,y_test = process_data()
    return X_train,y_train,X_val,y_val,X_test,y_test

def create_model(X_train,y_train,X_val,y_val,X_test,y_test):
    epochs = 25
    es_patience = 5
    lr_patience = 3
    dropout = None
    depth = 25
    nb_dense_block = 3
    nb_filter = 16
    growth_rate = 18
    bn = True
    reduction_ = 0.5
    bs = 32
    lr = 1E-2
    opt = {{choice([Adam(lr=1E-2), RMSprop(lr=1E-2),Adadelta(),Adamax(lr=1E-2),Nadam()])}}
    weight_file = 'hyperas_dn_lr_optimizer_wt_3Oct_1425.h5'
    nb_classes = 1
    img_dim = (2,96,96) 
    n_channels = 2 

    
    model  = DenseNet(depth=depth, nb_dense_block=nb_dense_block,
                 growth_rate=growth_rate, nb_filter=nb_filter,
                 dropout_rate=dropout,activation='sigmoid',
                 input_shape=img_dim,include_top=True,
                 bottleneck=bn,reduction=reduction_,
                 classes=nb_classes,pooling='avg',
                 weights=None)
    
    model.summary()
    model.compile(loss=binary_crossentropy, optimizer=opt, metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', patience=es_patience,verbose=1)
    checkpointer = ModelCheckpoint(filepath=weight_file,verbose=1, save_best_only=True)

    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=lr_patience, min_lr=0.5e-6,verbose=1)

    model.fit(X_train,y_train,
          batch_size=bs,
          epochs=epochs,
          callbacks=[lr_reducer,checkpointer,es],
          validation_data=(X_val,y_val),
          verbose=2)
    
    score, acc = model.evaluate(X_val,y_val)
    print("current val accuracy:%0.3f"%acc)
    pred = model.predict(X_val)
    auc_score = roc_auc_score(y_val,pred)
    print("current auc_score ------------------> %0.3f"%auc_score)

    model = load_model(weight_file) #This is the best model
    score, acc = model.evaluate(X_val,y_val)
    print("Best saved model val accuracy:%0.3f"% acc)
    pred = model.predict(X_val)
    auc_score = roc_auc_score(y_val,pred)
    print("best saved model auc_score ------------------> %0.3f"%auc_score)

    
    return {'loss': -auc_score, 'status': STATUS_OK, 'model': model}  

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--evals", type=int, default=40, help="no evaluations")
    args = vars(ap.parse_args())
    evals = args["evals"]
    print("no of  evals",evals)

    X_train,y_train,X_val,y_val,X_test,y_test = data()
    trials = Trials()
    try:
        best_run, best_model = optim.minimize(model=create_model, data=data,
        functions = [process_data],
        algo=tpe.suggest,max_evals=evals,trials=trials)

        print("best model",best_model)
        print("best run",best_run)
        print("Evalutation of best performing model:")
        print(best_model.evaluate(X_test, y_test,verbose=0))
        pred = best_model.predict(X_test,verbose=0)
        auc_score = roc_auc_score(y_test,pred)
        print("TEST roc_auc_score %0.3f"%auc_score)
        print("----------trials-------------")
	
	
    except Exception as e:
        print("Exception caught!")
        print(e)
        print(trials.trials)

    for i in trials.trials:
        vals = i.get('misc').get('vals')
        results = i.get('result').get('loss')
        print(vals,results)    

    K.clear_session()
