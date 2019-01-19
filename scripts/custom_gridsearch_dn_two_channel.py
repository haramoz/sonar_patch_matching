from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import random
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
from time import sleep
import timeit

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.datasets import cifar10
from keras.optimizers import RMSprop,Adam,Adadelta,Adamax,Nadam,SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense
#from keras_contrib.applications import DenseNet

import keras_densenet

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
from keras.models import model_from_json
from keras.models import load_model

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


import pickle
seed = 7
np.random.seed(seed)

def process_data():
    f = h5py.File('/home/amalli2s/thesis/keras/matchedImagesSplitClasses-2017-02-24-17-39-44-96-96-split-val0.15-tr0.7-tst0.15.hdf5','r')
    X_train = f['X_train'].value
    y_train = f['y_train'].value
    X_test = f['X_val'].value
    y_test = f['y_val'].value
    X_train_, X_val, y_train_, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)
    
    return X_train,y_train,X_val,y_val,X_test,y_test


def fit_model(data,layers,growth_rate,nb_dense_block,nb_filter,dropout,lr,epochs,opt,reduction,bn,batch_size,pooling):
    es_patience = 4    
    lr_patience = 3    
    weight_file = 'keras_densenet_simple_wt_13Dec2100.h5' 
    file_name = 'keras_densenet_twochannel_14Dec_0000'
    print("------------------------ current config for the test -------------------------")
    print("Layers: ",layers," Growth_rate: ",growth_rate," nb_filter: ",nb_filter," dropout: ",dropout)
    print("dense_block: ",nb_dense_block," reduction: ",reduction," bottleneck: ",bn)
    print("Epochs: ",epochs," batch_size: ",batch_size," lr: ",lr," optimizer: ",opt)
    print(" es_patience: ",es_patience," lr_patience: ",lr_patience," pooling: ",pooling)
    print("------------------------	  end of configs        -------------------------")

    img_dim = (2,96,96)
    classes = 1 #for binary classification

    """input_shape=None, depth=40, nb_dense_block=3,growth_rate=12, nb_filter=-1,nb_layers_per_block=-1,
             bottleneck=False, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4,subsample_initial_block=False,
             include_top=True,weights=None,input_tensor=None,pooling=None,classes=10,
             activation='softmax',transition_pooling='avg'"""

    model=keras_densenet.DenseNet(nb_dense_block=nb_dense_block,
                     growth_rate=growth_rate,
                     nb_filter=nb_filter,
                     nb_layers_per_block=layers,
                     dropout_rate=dropout,
                     input_shape=img_dim,
                     include_top=True,
                     classes=classes,
                     pooling=pooling,
                     bottleneck=bn,
                     reduction=reduction,
                     activation='sigmoid',
                     weights=None,
                     subsample_initial_block=True,
                     transition_pooling='avg')    

    if opt=='adam':
        optimizer = Adam(lr=lr)
    elif opt=='nadam':
        optimizer=Nadam(lr=lr)
    elif opt=='adadelta':        
        optimizer=Adadelta(lr=lr)
    elif opt=='adamax':        
        optimizer=Adamax(lr=lr)
    elif opt=='rmsprop':        
        optimizer=RMSprop(lr=lr)
    else:
        optimizer=SGD(lr=lr,momentum=0.9)


    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
    print('Finished compiling')

    #model.compile(loss=binary_crossentropy, optimizer=opt, metrics=['accuracy'])

    es = EarlyStopping(monitor='val_acc', patience=es_patience,verbose=1)
    checkpointer = ModelCheckpoint(filepath=weight_file, verbose=2, save_best_only=True)
    
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),
                               cooldown=0, patience=lr_patience, min_lr=0.5e-6,verbose=1)

    model.fit(data[0],data[1],
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(data[2], data[3]),
          callbacks=[es,lr_reducer],
          verbose=2)
    #model = load_model(weight_file) #This is the best model

    score, acc = model.evaluate(data[4], data[5], verbose=0)
    print("Test accuracy:%0.3f"% acc)
    pred = model.predict(data[4])
    auc_score = roc_auc_score(data[5],pred)
    auc_score = np.round(auc_score,4)
    print("current auc_score ------------------> %0.3f"%auc_score)
    if(auc_score > .95):
        model_json = model.to_json()
        score = str(auc_score)
        with open(file_name+score+".json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(file_name+score+".h5")
        print("Saved model to disk")

    del model
    K.clear_session()
    return acc,auc_score

def process_fit(config):
    start_time = timeit.default_timer()
    data = process_data()
    #nb_layers_per_block,growth_rate,nb_dense_block,nb_filter,dropout,lr,epochs,opt,reduction,bn,pooling
    layers = []
    temp_layers = config[0].split("-")
    for i in temp_layers:
        layers.append(int(i))
    growth_rate=int(config[1])
    nb_dense_block=int(config[2])
    nb_filter=int(config[3])
    dropout=float(config[4])
    lr=float(config[5])
    epochs=int(config[6])
    opt=config[7]

    trials = 3
    reduction=float(config[8])
    if config[9]=='FALSE':
        bn=False
    else:
        bn=True
    batch_size=int(config[10])
    pooling=config[11]

    accs = []
    aucs = []
    for i in range(trials):
        acc,auc = fit_model(data,layers,growth_rate,nb_dense_block,nb_filter,dropout,lr,epochs,opt,reduction,bn,batch_size,pooling)
        accs.append(acc)
        aucs.append(auc)
    print("accuracies: ",accs)
    print("aucs: ",aucs)
    mean = np.round(np.mean(aucs),3)
    std = np.round(np.std(aucs),3)
    result = str(mean)+'+/-'+str(std)
    max = np.max(aucs)
    print("mean and std AUC: ",result," max:  ",max)
    return result,max

if __name__ == '__main__':
    start_time = timeit.default_timer()    
    search_space = []
    #accuracies = []
    auc_scores = []
    max_aucs = []
    with open('./search_spaces/dn_two_channel_13Dec.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                print(row)
                search_space.append(row)    
                auc,max=process_fit(row)
                auc_scores.append(auc)
                max_aucs.append(np.round(max,3))
    #print(auc_scores)
    #print(max_aucs)
    #print(search_space)
    if(len(auc_scores) == len(search_space)):
        results = zip(search_space,auc_scores,max_aucs)
        for i in results:
            print(i)
    else:
        print(auc_scores)
    K.clear_session()
    print(timeit.default_timer() - start_time)
