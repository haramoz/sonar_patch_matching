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
from keras.callbacks import EarlyStopping,ModelCheckpoint

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

import densenet

import pickle

def process_data():
    filename = 'sonar_data'
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

    y_test_inverted = 1 - y_test
    y_train_inverted = 1 - y_train
    y_val_inverted = 1 - y_val

    return x_train1, x_train2, y_train, x_val1, x_val2, y_val, x_test1, x_test2, y_test


def data():
    x_train1, x_train2, y_train, x_val1, x_val2, y_val, x_test1, x_test2, y_test = process_data()
    return x_train1, x_train2, y_train, x_val1, x_val2, y_val, x_test1, x_test2, y_test



def create_base_network(img_dim,depth,nb_filter,nb_dense_block,growth_rate,dropout1,weight_decay):
    '''Base network to be shared (eq. to feature extraction).
    '''    
    #default values
    nb_classes = 2 
    img_dim = (1,96,96) 
    n_channels = 1 

    #depth =  7 #40 #3N+4
    #nb_filter = 16 #16 #increasing this worsens the result
    #nb_dense_block = 1 #3
    #growth_rate = 12
    #dropout_rate = 0.3

    net = densenet.DenseNet(nb_classes,
                  img_dim,
                  depth,
                  nb_dense_block,
                  growth_rate,
                  nb_filter,
                  dropout_rate=dropout1,
                  weight_decay=weight_decay)
    
    return net

def create_base_network_vgg(input_shape,dense_filter,dense_filter1,dense_filter2,dropout1,dropout2,layers):
    random_seed = 7

    model = Input(shape=input_shape)
    x = Conv2D(16, (3, 3), padding="same",activation='relu', data_format='channels_first',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=random_seed),
              bias_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=random_seed))(model)

    x = Conv2D(16, (3, 3), padding="same", activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=random_seed),
              bias_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=random_seed))(x)
    x = MaxPooling2D(pool_size=(2, 2),strides=(2, 2))(x)

    x = Conv2D(32, (3, 3), padding="same", activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=random_seed),
              bias_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=random_seed))(x)

    x = Conv2D(32, (3, 3), padding="same", activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=random_seed),
              bias_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=random_seed))(x)

    x = MaxPooling2D(pool_size=(2, 2),strides=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(dense_filter, kernel_initializer=keras.initializers.he_normal(seed=random_seed),activation='relu')(x)
    if layers == "two":
        x = Dense(dense_filter1, kernel_initializer=keras.initializers.he_normal(seed=random_seed),activation='relu')(x)
        x = Dropout(dropout1)(x)
    elif layers == "three":
        x = Dense(dense_filter2, kernel_initializer=keras.initializers.he_normal(seed=random_seed),activation='relu')(x)
        x = Dropout(dropout2)(x)
    return Model(model,x)


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def contrastive_loss_altered(y_true, y_pred):
    margin = 1.0
    return K.mean((1-y_true) * K.square(y_pred) +
                  y_true * K.square(K.maximum(margin - y_pred, 0)))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def create_model(x_train1, x_train2, y_train,x_val1, x_val2, y_val, x_test1, x_test2, y_test):
    epochs = 7
    input_shape = (1,96,96)
    patience_ = 1
    dense_filter = 512
    dense_filter1 = 512
    dropout = {{uniform(0,1)}}
    dropout1 = {{uniform(0,1)}}
    dropout2 = {{uniform(0,1)}}
    depth = 7
    nb_dense_block = 1
    nb_filter = 16
    growth_rate = 12
    weight_decay = 1E-4
    lr = 5E-5

    base_network = create_base_network(input_shape,depth,nb_filter,nb_dense_block,growth_rate,dropout1,weight_decay)
    use_distance = False

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    combined_features = concatenate([processed_a, processed_b], name = 'merge_features')

    combined_features = Dense(dense_filter, kernel_initializer=keras.initializers.he_normal())(combined_features)
    combined_features = Dropout(dropout)(combined_features)
    combined_features = Dense(dense_filter, kernel_initializer=keras.initializers.he_normal())(combined_features)
    combined_features = Activation('relu')(combined_features)
    #combined_features = BatchNormalization()(combined_features)
    combined_features = Dropout(dropout2)(combined_features)
    combined_features = Dense(1, activation = 'sigmoid')(combined_features)

    model = Model(inputs = [input_a, input_b], outputs = [combined_features], name = 'model')

    model.summary()
    opt = Adam(lr=lr)
    #opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    model.compile(loss=binary_crossentropy, optimizer=opt, metrics=['accuracy'])

    es = EarlyStopping(monitor='val_acc', patience=patience_,verbose=1)
    checkpointer = ModelCheckpoint(filepath='keras_weights.hdf5',
                                   verbose=1,
                                   save_best_only=True)
    model.fit([x_train1, x_train2],y_train,
          batch_size=64,
          epochs=epochs,
          validation_data=([x_val1,x_val2], y_val),
          callbacks=[es],
          verbose=0)

    score, acc = model.evaluate([x_test1, x_test2], y_test,verbose=0)
    print('Test accuracy:', acc)
    pred = model.predict([x_test1, x_test2],verbose=0)


    threshold = 0.4
    pred_scores2 = (pred>threshold).astype(int) 

    test_acc2 = accuracy_score(y_test,pred_scores2)

    auc_score = roc_auc_score(y_test,pred)
    print("score new --------------- >",test_acc2)
    print("auc_score ------------------> ",auc_score)


    return {'loss': -auc_score, 'status': STATUS_OK, 'model': model}




if __name__ == '__main__':

    x_train1, x_train2, y_train,x_val1, x_val2, y_val, x_test1, x_test2, y_test = data()
    trials = Trials()
    best_run, best_model = optim.minimize(model=create_model, data=data,
    functions = [process_data,create_base_network,euclidean_distance,eucl_dist_output_shape,contrastive_loss_altered],
    algo=tpe.suggest,max_evals=100,trials=trials)
    print("best model",best_model)
    print("best run",best_run)
    print("Evalutation of best performing model:")
    print(best_model.evaluate([x_test1, x_test2], y_test,verbose=0))
    pred = best_model.predict([x_test1, x_test2],verbose=0)
    auc_score = roc_auc_score(y_test,pred)
    print("val roc_auc_score %0.3f"%auc_score)
    print("----------trials-------------")
    for i in trials.trials:
        vals = i.get('misc').get('vals')
        results = i.get('result').get('loss')
        print(vals,results)    
    K.clear_session()


