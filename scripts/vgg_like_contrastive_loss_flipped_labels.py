from __future__ import absolute_import
from __future__ import print_function
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
from keras.callbacks import EarlyStopping

from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import pickle

random_seed = 7

def load_data():
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
    return (x_train1, x_train2, y_train), (x_val1, x_val2, y_val), (x_test1, x_test2, y_test)
    
(x_train1, x_train2, y_train), (x_val1, x_val2, y_val), (x_test1, x_test2, y_test) = load_data() 

y_test_inverted = 1 - y_test
y_train_inverted = 1 - y_train
y_val_inverted = 1 - y_val

    
def create_base_network_vgg(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    model = Input(shape=input_shape)
    x = Conv2D(16, (3, 3), padding="same",activation='relu', data_format='channels_first', 
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=random_seed), 
               bias_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=random_seed))(model)
    
    x = Conv2D(16, (3, 3), padding="same", activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=random_seed), 
               bias_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=random_seed))(x)    
    x = MaxPooling2D(pool_size=(2, 2),strides=(2, 2))(x)
    #x = Dropout(0.2)(x)
    
    x = Conv2D(32, (3, 3), padding="same", activation='relu', 
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=random_seed),
               bias_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=random_seed))(x)
    
    x = Conv2D(32, (3, 3), padding="same", activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=random_seed), 
               bias_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=random_seed))(x)
    x = MaxPooling2D(pool_size=(2, 2),strides=(2, 2))(x)#if stride==None default to pool_size
    #x = Dropout(0.5)(x)
    
    x = Conv2D(64, (3, 3), padding="same", activation='relu', 
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=random_seed),
               bias_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=random_seed))(x)
    
    x = Conv2D(64, (3, 3), padding="same", activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=random_seed), 
               bias_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=random_seed))(x)
    
    x = Conv2D(64, (3, 3), padding="same", activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=random_seed), 
               bias_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=random_seed))(x)
    x = MaxPooling2D(pool_size=(2, 2),strides=(2, 2))(x)#if stride==None default to pool_size
    
    x = Flatten()(x)
    #x = Dense(500, activation='relu')(x)
    x = Dense(2048, kernel_initializer=keras.initializers.lecun_normal(seed=random_seed),activation='relu')(x)
    #x = Dropout(0.1)(x)
    x = Dense(2048, kernel_initializer=keras.initializers.lecun_normal(seed=random_seed),activation='relu')(x)
    #x = Dropout(0.1)(x)
    x = Dense(2048, kernel_initializer=keras.initializers.lecun_normal(seed=random_seed),activation='relu')(x)
    #x = Dropout(0.1)(x)

    #model = Dense(num_classes)(model)    
    return Model(model,x)

def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)

def plot_auc(fpr,tpr):
    plt.figure()
    plt.plot(fpr, tpr,'m-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.show()

def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def acc(y_true, y_pred):
    ones = K.ones_like(y_pred)
    return K.mean(K.equal(y_true, ones - K.clip(K.round(y_pred), 0, 1)), axis=-1)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    loss = K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
    return loss

#When the labels are flipped then this function should be used. It conforms more to the formula from the paper
#missing a 0.5 multiplication though. Does it matter?
def contrastive_loss_altered(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1.0
    return K.mean((1-y_true) * K.square(y_pred) +
                  y_true * K.square(K.maximum(margin - y_pred, 0)))
    
batch_size = 128
epochs = 2
input_shape = (1,96,96)
num_classes = 2
is_using_distance = True #IMPORTANT
init_lr = 0.00001 #1e-3 #1e-4 #1e-2=0.01
init_decay = 1e-6 #1e-5 
patience_ = 2

#(x_train1, x_train2, y_train), (x_val1, x_val2, y_val), (x_test1, x_test2, y_test) = load_data() 

base_network = create_base_network_vgg(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

processed_a = base_network(input_a)
processed_b = base_network(input_b)

opt = Adam(lr=init_lr, decay=init_decay)

if  is_using_distance:
    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    model = Model([input_a, input_b], distance)
    model.compile(loss=contrastive_loss_altered, optimizer=opt, metrics=["accuracy"])
    model.summary()
    #model.compile(loss=contrastive_loss, optimizer=opt, metrics=[acc])
else:
    combined_features = concatenate([processed_a, processed_b], name = 'merge_features')
    #combined_features = Dense(64)(combined_features)
    #combined_features = BatchNormalization()(combined_features)
    #combined_features = Activation('relu')(combined_features)
    combined_features = Dense(512, kernel_initializer=keras.initializers.lecun_normal(seed=random_seed))(combined_features)
    combined_features = BatchNormalization()(combined_features)
    combined_features = Activation('relu')(combined_features)

    '''combined_features = Dense(64)(combined_features)
    combined_features = BatchNormalization()(combined_features)
    combined_features = Activation('relu')(combined_features)

    combined_features = Dense(16)(combined_features)
    combined_features = BatchNormalization()(combined_features)
    combined_features = Activation('relu')(combined_features)'''

    combined_features = Dense(1, activation = 'sigmoid')(combined_features)

    model = Model(inputs = [input_a, input_b], outputs = [combined_features], name = 'Similarity_Model')
    
    model.summary()

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
es = EarlyStopping(monitor='val_loss', patience=patience_,verbose=1)

model.fit([x_train1,x_train2],y_train_inverted,
      batch_size=batch_size,
      epochs=epochs,
      validation_data=([x_val1,x_val2], y_val_inverted),
      callbacks=[es],
      verbose=1)

#y_test = 1 - y_test """You use this once and the  values gets flipped forever!!"""
scores1 = model.evaluate([x_test1, x_test2], y_test_inverted)
print("test accuracy : ",scores1)
pred = model.predict([x_test1, x_test2])
if  is_using_distance:
    fpr, tpr, thresholds = metrics.roc_curve(y_test_inverted, pred)
    #scores2 = compute_accuracy(y_test_inverted, pred) #not required since the 
    #print("test accuracy %0.3f"%scores2)
else:
    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)
#plot_auc(fpr,tpr)

if  is_using_distance:
    auc_score = roc_auc_score(y_test_inverted, pred)
else:
    auc_score = roc_auc_score(y_test, pred)
print("roc_auc_score %0.3f"%auc_score)
