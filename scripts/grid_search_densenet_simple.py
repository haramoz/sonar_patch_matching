# Use scikit-learn to grid search the batch size and epochs
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
from keras.models import load_model
import keras.initializers
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from keras_contrib.applications import DenseNet


def create_model():
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def process_data():
    random_seed = 7

    f = h5py.File('matchedImagesSplitClasses-2017-02-24-17-39-44-96-96-split-val0.15-tr0.7-tst0.15.hdf5','r')
    X_train = f['X_train'].value
    y_train = f['y_train'].value
    X_test = f['X_val'].value
    y_test = f['y_val'].value
    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_seed)
 
    #return X_train,y_train,X_val,y_val,X_test,y_test
    return X_train,y_train,X_test,y_test
# Function to create model, required for KerasClassifier
def create_model():
    epochs = 40
    es_patience = 7
    lr_patience = 5
    dropout = None
    depth = {{choice([7,13,19,25,31])}}
    nb_dense_block = {{choice([2,3])}}
    nb_filter = 16
    growth_rate = {{choice([6,10,14,18])}}
    bn = True
    reduction_ = 0.5
    bs = 32
    lr = 1E-4 #########################################################CHange file name##########################################
    weight_file = 'keras_densenet_simple_wt_29Sept_2200.h5'
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
    opt = Adam(lr=lr)
    model.compile(loss=binary_crossentropy, optimizer=opt, metrics=['accuracy'])

    """es = EarlyStopping(monitor='val_loss', patience=es_patience,verbose=1)
    #es = EarlyStopping(monitor='val_acc', patience=es_patience,verbose=1,restore_best_weights=True)
    checkpointer = ModelCheckpoint(filepath=weight_file,verbose=1, save_best_only=True)

    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=lr_patience, min_lr=0.5e-6,verbose=1)

    model.fit(X_train,y_train,
          batch_size=bs,
          epochs=epochs,
          callbacks=[es,lr_reducer,checkpointer],
          validation_data=(X_val,y_val),
          verbose=2)
    
    score, acc = model.evaluate(X_test, y_test)
    print('current Test accuracy:', acc)
    pred = model.predict(X_test)
    auc_score = roc_auc_score(y_test,pred)
    print("current auc_score ------------------> ",auc_score)"""

    """model = load_model(weight_file) #This is the best model
    score, acc = model.evaluate(X_test, y_test)
    print('Best saved model Test accuracy:', acc)
    pred = model.predict(X_test)
    auc_score = roc_auc_score(y_test,pred)
    print("best saved model auc_score ------------------> ",auc_score)"""
    
    return model 

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
X_train,y_train,X_test,y_test = process_data()

# create model
model = KerasClassifier(build_fn=create_model, verbose=0)
# define the grid search parameters
batch_size = [10, 20]
epochs = [10,20]
optimizer = ['RMSprop', 'Adam']
#param_grid = dict(optimizer=optimizer)
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1,verbose=2)

grid_result = grid.fit(X_train,y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
