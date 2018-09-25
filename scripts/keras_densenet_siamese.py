from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import random
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.datasets import cifar10
from keras.optimizers import Adam
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

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score


import pickle
seed = 7
np.random.seed(seed)

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


def data():
    x_train1, x_train2, y_train, x_val1, x_val2, y_val, x_test1, x_test2, y_test = process_data()
    return x_train1, x_train2, y_train, x_val1, x_val2, y_val, x_test1, x_test2, y_test



"""def process_data():
    f = h5py.File('matchedImagesSplitClasses-2017-02-24-17-39-44-96-96-split-val0.15-tr0.7-tst0.15.hdf5','r')
    ln_training = 39840
    X_train = f['X_train'].value
    #X_train_resize = X_train[0:ln_training,:,:,:]
    #print(X_train_resize.shape)
    y_train = f['y_train'].value
    #y_train_resize = y_train[0:ln_training,]
    #y_train_categorical = np_utils.to_categorical(y_train_resize, 2)
    #X_reshaped = X_train_resize.reshape(*X_train_resize.shape[:1], -2)
    #print(X_reshaped.shape)
    ln_validation = 7440
    X_val = f['X_val'].value
    #X_val_resize = X_val[0:ln_validation,:,:,:]

    #X_val_reshaped = X_val
    #X_val_reshaped = X_val_resize.reshape(*X_val_resize.shape[:1], -2)

    y_val = f['y_val'].value
    #y_val_reshaped = y_val[0:ln_validation]
    #y_val_categorical = np_utils.to_categorical(y_val_reshaped, 2)

    X_train = np.moveaxis(X_train, 1, -1)
    X_val = np.moveaxis(X_val, 1, -1)
    print(X_train.shape)
    print(X_val.shape)
    return X_train,y_train,X_val,y_val"""

def create_base_network():
    '''Base network to be shared (eq. to feature extraction).
    '''    
    batch_size = 8
    nb_classes = 2
    epochs = 30

    img_rows, img_cols = 96, 96
    img_channels = 1
    print(K.image_data_format())

    # Parameters for the DenseNet model builder
    img_dim = (img_channels, img_rows, img_cols) if K.image_data_format() == 'channels_first' else (img_rows, img_cols, img_channels)

    depth = 7
    nb_dense_block = 2
    growth_rate = 6
    nb_filter = 16
    dropout_rate = 0.0  # 0.0 for data augmentation
    classes = 2 #1

    """input_shape=None, depth=40, nb_dense_block=3,growth_rate=12, nb_filter=-1,nb_layers_per_block=-1,
             bottleneck=False, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4,subsample_initial_block=False,
             include_top=True,weights=None,input_tensor=None,pooling=None,classes=10,
             activation='softmax',transition_pooling='avg'"""

    net = keras_densenet.DenseNet(depth=depth, nb_dense_block=nb_dense_block,
                     growth_rate=growth_rate, nb_filter=nb_filter,
                     dropout_rate=dropout_rate,
                     input_shape=img_dim,
                     include_top=False,
                     classes=classes,
                     pooling='avg',bottleneck=True,reduction=0.5,
                     #activation='sigmoid'
                     weights=None)


    
    return net

def fit_model(x_train1, x_train2, y_train,x_val1, x_val2, y_val, x_test1, x_test2, y_test):
    epochs = 15
    input_shape = (1,96,96)
    lr = 1E-3
    patience_ = 3    
    batch_size = 4

    base_network = create_base_network()
    use_distance = False

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    combined_features = concatenate([processed_a, processed_b], name = 'merge_features')
    """combined_features = Dense(1024, kernel_initializer=keras.initializers.he_normal(),activation='relu')(combined_features)
    combined_features = Dropout(dropout)(combined_features)"""

    combined_features = Dense(512, kernel_initializer=keras.initializers.he_normal())(combined_features)    
    combined_features = Activation('relu')(combined_features)
    combined_features = BatchNormalization()(combined_features)
    #combined_features = Dropout(0.5)(combined_features)
    combined_features = Dense(1, activation = 'sigmoid')(combined_features)

    model = Model(inputs = [input_a, input_b], outputs = [combined_features], name = 'model')
    model.summary()

    optimizer = Adam(lr=lr)  # Using Adam instead of SGD to speed up training
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
    print('Finished compiling')

    #model.compile(loss=binary_crossentropy, optimizer=opt, metrics=['accuracy'])

    es = EarlyStopping(monitor='val_acc', patience=patience_,verbose=1,restore_best_weights=True)
    checkpointer = ModelCheckpoint(filepath='keras_densenet_siamese_2509_weights.hdf5', verbose=2, save_best_only=True)
    
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),
                               cooldown=0, patience=5, min_lr=0.5e-6,verbose=1)

    model.fit([x_train1, x_train2],y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=([x_val1,x_val2], y_val),
          callbacks=[es,lr_reducer],
          verbose=1)

    score, acc = model.evaluate([x_test1, x_test2], y_test)
    print('Test accuracy:', acc)



    """Y_train = np_utils.to_categorical(trainY, nb_classes)
    Y_test = np_utils.to_categorical(testY, nb_classes)

    generator = ImageDataGenerator(rotation_range=15,
                                   width_shift_range=5. / 32,
                                   height_shift_range=5. / 32)

    generator.fit(trainX, seed=0)

    weights_file = 'DenseNet-40-12-CIFAR-10.h5'

    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),
                                   cooldown=0, patience=10, min_lr=0.5e-6)
    early_stopper = EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=20, restore_best_weights=True)

    model_checkpoint = ModelCheckpoint(weights_file, monitor='val_acc', save_best_only=True,
                                       save_weights_only=True, mode='auto')

    callbacks = [lr_reducer, early_stopper, model_checkpoint]

    model.fit_generator(generator.flow(trainX, Y_train, batch_size=batch_size), steps_per_epoch=len(trainX) // batch_size,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=(testX, Y_test),
                        verbose=2)

    scores = model.evaluate(testX, Y_test, batch_size=batch_size)
    print('Test loss : ', scores[0])
    print('Test accuracy : ', scores[1])"""

if __name__ == '__main__':
    
    x_train1, x_train2, y_train,x_val1, x_val2, y_val, x_test1, x_test2, y_test = data()
    fit_model(x_train1, x_train2, y_train,x_val1, x_val2, y_val, x_test1, x_test2, y_test)
