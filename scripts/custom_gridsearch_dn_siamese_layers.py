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


import pickle
seed = 7
np.random.seed(seed)

def process_data():
    filename = '/home/amalli2s/thesis/keras/sonar_data'
    infile = open(filename,'rb')
    sonar_data_dict = pickle.load(infile)

    infile.close()
    x_train1 = sonar_data_dict.get("x_train1")
    x_train2 = sonar_data_dict.get("x_train2")
    y_train = sonar_data_dict.get("y_train")

    x_val1 = sonar_data_dict.get("x_val1")
    x_val2 = sonar_data_dict.get("x_val2")
    y_val = sonar_data_dict.get("y_val")

    x_test1 = sonar_data_dict.get("x_test1")
    x_test2 = sonar_data_dict.get("x_test2")
    y_test = sonar_data_dict.get("y_test")

    #y_test_inverted = 1 - y_test
    #y_train_inverted = 1 - y_train
    #y_val_inverted = 1 - y_val
    """x_train1 = np.moveaxis(x_train1, 1, -1)
    x_train2 = np.moveaxis(x_train2, 1, -1)

    x_val1 = np.moveaxis(x_val1, 1, -1)
    x_val2 = np.moveaxis(x_val2, 1, -1)

    x_test1 = np.moveaxis(x_test1, 1, -1)
    x_test2 = np.moveaxis(x_test2, 1, -1)"""


    return x_train1, x_train2, y_train, x_val1, x_val2, y_val, x_test1, x_test2, y_test


def create_base_network(layers,growth_rate,nb_dense_block,nb_filter,dropout_rate,reduction_,bn):
    '''Base network to be shared (eq. to feature extraction).
    '''    
    nb_classes = 2
    img_rows, img_cols = 96, 96
    img_channels = 1

    # Parameters for the DenseNet model builder
    img_dim = (img_channels, img_rows, img_cols) if K.image_data_format() == 'channels_first' else (img_rows, img_cols, img_channels)

    #depth = 22
    #nb_dense_block = 4
    #growth_rate = 30
    #nb_filter = 32
    #dropout_rate = 0.2  # 0.0 for data augmentation
    classes = 2 #1
    #reduction_ = 0.5
    #bn = True
    print("------------------------ current config for the test -------------------------")
    print("Layers: ",layers," Growth_rate: ",growth_rate," nb_filter: ",nb_filter," dropout: ",dropout_rate)
    print("dense_block ",nb_dense_block," reduction_: ",reduction_," bottleneck: ",bn)
    print("------------------------	  end of configs        -------------------------")

    """input_shape=None, depth=40, nb_dense_block=3,growth_rate=12, nb_filter=-1,nb_layers_per_block=-1,
             bottleneck=False, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4,subsample_initial_block=False,
             include_top=True,weights=None,input_tensor=None,pooling=None,classes=10,
             activation='softmax',transition_pooling='avg'"""

    net =keras_densenet.DenseNet(nb_dense_block=nb_dense_block,
                     growth_rate=growth_rate, nb_filter=nb_filter,
                     nb_layers_per_block=layers,
                     dropout_rate=dropout_rate,
                     input_shape=img_dim,
                     include_top=False,
                     classes=classes,
                     pooling='flatten',
                     bottleneck=bn,
                     reduction=reduction_,
                     #activation='sigmoid'
                     weights=None,
                     subsample_initial_block=True,
                     transition_pooling='max')


    
    return net

def fit_model(data,depth,growth_rate,nb_dense_block,nb_filter,dropout,lr,epochs,opt,reduction,bn,batch_size,fc_dropout,fc_filter,fc_layers):
    input_shape = (1,96,96)
    es_patience = 4    
    lr_patience = 3    
    #batch_size = 64
    weight_file = 'keras_densenet_siamese_8Nov_2300_weights.h5' 
    file_name = 'keras_densenet_siamese_8Nov_2300' 
    dense_dropout = 0.5
    print("Epochs ",epochs," batch_size: ",batch_size," lr: ",lr," optimizer: ",opt)
    print(" es_patience: ",es_patience," lr_patience: ",lr_patience)
    print(" batch_size: ",batch_size," fc_dropout: ",fc_dropout," fc_filter: ",fc_filter," fc_layers: ",fc_layers)

    base_network = create_base_network(depth,growth_rate,nb_dense_block,nb_filter,dropout,reduction,bn)
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    combined_features = concatenate([processed_a, processed_b], name = 'merge_features')

    combined_features = Dense(fc_filter, kernel_initializer=keras.initializers.he_normal())(combined_features)    
    combined_features = Activation('relu')(combined_features)
    combined_features = BatchNormalization()(combined_features)
    combined_features = Dropout(fc_dropout)(combined_features)

    combined_features = Dense(1, activation = 'sigmoid')(combined_features)

    model = Model(inputs = [input_a, input_b], outputs = [combined_features], name = 'model')
    model.summary()
    if opt=='adam':
        optimizer = Adam(lr=lr)  # Using Adam instead of SGD to speed up training
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

    model.fit([data[0], data[1]],data[2],
          batch_size=batch_size,
          epochs=epochs,
          validation_data=([data[3], data[4]],data[5]),
          callbacks=[es,lr_reducer],
          verbose=2)
    #model = load_model(weight_file) #This is the best model

    score, acc = model.evaluate([data[6], data[7]],data[8], verbose=0)
    print("Test accuracy:%0.3f"% acc)
    pred = model.predict([data[6], data[7]])
    auc_score = roc_auc_score(data[8],pred)
    auc_score = np.round(auc_score,4)
    print("current auc_score ------------------> %0.3f"%auc_score)
    if(auc_score > .94):
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
    data = process_data()
    #nb_layers_per_block,growth_rate,nb_dense_block,nb_filter,dropout,lr,epochs,opt,reduction,bn,fc_dropout,fc_filter,fc_layers
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
    reduction=float(config[8])
    if config[9]=='FALSE':
        bn=False
    else:
        bn=True
    batch_size=int(config[10])
    fc_dropout=float(config[11])
    fc_filter=int(config[12])
    fc_layers=int(config[13])

    accs = []
    aucs = []
    for i in range(3):
        acc,auc = fit_model(data,layers,growth_rate,nb_dense_block,nb_filter,dropout,lr,epochs,opt,reduction,bn,batch_size,fc_dropout,fc_filter,fc_layers)
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
    with open('./search_spaces/dn_siamese_random_27oct.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                #print(row)
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
