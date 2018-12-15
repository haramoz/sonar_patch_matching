from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import random
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
import csv

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
import keras.initializers
from keras.callbacks import EarlyStopping,ModelCheckpoint, ReduceLROnPlateau

from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import pickle

random_seed = 7

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

    y_test_inverted = 1 - y_test
    y_train_inverted = 1 - y_train
    y_val_inverted = 1 - y_val

    return x_train1, x_train2, y_train_inverted, x_val1, x_val2, y_val_inverted, x_test1, x_test2, y_test_inverted

def set_initializer(init):
    init = init.strip()
    if init == "he_normal":
        initializer = keras.initializers.he_normal(seed=None)
        #initializer = keras.initializers.he_normal(seed=random_seed)
    elif init == "he_uniform":
        initializer = keras.initializers.he_uniform(seed=None)
    elif init == "lecun_normal":
        initializer = keras.initializers.lecun_normal(seed=None)
    elif init == "lecun_uniform":
        initializer = keras.initializers.lecun_uniform(seed=None)
    elif init == "glorot_normal":
        initializer = keras.initializers.glorot_normal(seed=None)
    elif init == "glorot_uniform":
        initializer = keras.initializers.glorot_uniform(seed=None)
    elif init == "random_uniform":
        initializer = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
    else:
        initializer = RandomNormal(mean=0.0, stddev=0.05, seed=random_seed) #random_normal
    return initializer

def create_base_network_vgg(input_shape,conv2d_filters,kernel_sizes,initialization,layers,dense_filter1,dense_filter2,dense_filter3,dropout1,dropout2,dropout3,use_bn,init,init_dense):

    print("conv2d_filters: ",conv2d_filters," kernel_sizes: ",kernel_sizes," initialization: ",initialization," layers: ",layers)

    print("dense_filter1: ",dense_filter1," dense_filters2: ",dense_filter2," dense_filters3: ",dense_filter3)

    print("dropout1: ",dropout1," dropout2: ",dropout2," dropout3: ",dropout3," use_bn: ",use_bn)

    print(" initializer: ",init," init_dense: ",init_dense)

    print("------------------------	  end of configs        -------------------------")
    
    initializer = set_initializer(init)
    init_dense = set_initializer(init_dense)

    model = Input(shape=input_shape)

    x = Conv2D(conv2d_filters[0], (kernel_sizes,kernel_sizes), padding="same",activation='relu', data_format='channels_first',
               kernel_initializer=initializer)(model)

    x = Conv2D(conv2d_filters[1], (kernel_sizes,kernel_sizes), padding="same", activation='relu',
              kernel_initializer=initializer)(x)

    x = MaxPooling2D(pool_size=(2, 2),strides=(2, 2))(x)

    x = Conv2D(conv2d_filters[2], (kernel_sizes,kernel_sizes), padding="same", activation='relu',
              kernel_initializer=initializer)(x)

    x = Conv2D(conv2d_filters[3], (kernel_sizes,kernel_sizes), padding="same", activation='relu',
              kernel_initializer=initializer)(x)

    x = MaxPooling2D(pool_size=(2, 2),strides=(2, 2))(x)

    x = Conv2D(conv2d_filters[3]*2, (kernel_sizes,kernel_sizes), padding="same", activation='relu',
              kernel_initializer=initializer)(x)
    x = Conv2D(conv2d_filters[3]*2, (kernel_sizes,kernel_sizes), padding="same", activation='relu',
              kernel_initializer=initializer)(x)
    x = Conv2D(conv2d_filters[3]*2, (kernel_sizes,kernel_sizes), padding="same", activation='relu',
              kernel_initializer=initializer)(x)
    x = MaxPooling2D(pool_size=(2, 2),strides=(2, 2))(x)

    x = Conv2D(conv2d_filters[3]*4, (kernel_sizes,kernel_sizes), padding="same", activation='relu',
              kernel_initializer=initializer)(x)
    x = Conv2D(conv2d_filters[3]*4, (kernel_sizes,kernel_sizes), padding="same", activation='relu',
              kernel_initializer=initializer)(x)
    x = Conv2D(conv2d_filters[3]*4, (kernel_sizes,kernel_sizes), padding="same", activation='relu',
              kernel_initializer=initializer)(x)

    x = MaxPooling2D(pool_size=(2, 2),strides=(2, 2))(x)

    x = Conv2D(conv2d_filters[3]*4, (kernel_sizes,kernel_sizes), padding="same", activation='relu',
              kernel_initializer=initializer)(x)
    x = Conv2D(conv2d_filters[3]*4, (kernel_sizes,kernel_sizes), padding="same", activation='relu',
              kernel_initializer=initializer)(x)
    x = Conv2D(conv2d_filters[3]*4, (kernel_sizes,kernel_sizes), padding="same", activation='relu',
              kernel_initializer=initializer)(x)

    x = MaxPooling2D(pool_size=(2, 2),strides=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(dense_filter1, kernel_initializer=init_dense, activation='relu')(x)
    if use_bn:        
        x = BatchNormalization()(x)
    x = Dropout(dropout1)(x)
    if layers == "two" or layers == "three":
        x = Dense(dense_filter2, kernel_initializer=init_dense, activation='relu')(x)
        if use_bn:        
            x = BatchNormalization()(x)
        x = Dropout(dropout2)(x)
    if layers == "three":
        x = Dense(dense_filter3, kernel_initializer=init_dense, activation='relu')(x)
        if use_bn:        
            x = BatchNormalization()(x)
        x = Dropout(dropout3)(x)
    return Model(model,x)


def plot_auc(fpr,tpr):
    plt.figure()
    plt.plot(fpr, tpr,'m-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.show()

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

#fit the model and process all the hyper parameters
def fit_model(data,conv2d_filters,kernel_sizes,initialization,layers,dense_filter1,dense_filter2,dense_filter3,dropout1,dropout2,dropout3,use_bn,batch_size,opt,lr,epochs,loss,initializer,init_dense):
    input_shape = (1,96,96)
    num_classes = 2
    is_using_distance = True #IMPORTANT
    init_decay = 1e-6 #1e-5
    es_patience = 4    
    lr_patience = 3    
    weight_file = 'keras_contrastive_loss_19Nov_weights.h5' 
    file_name = 'keras_contrastive_loss_19Nov'
    print("batch_size: ",batch_size," opt: ",opt," lr: ",lr," epochs: ",epochs," loss: ",loss)
    base_network = create_base_network_vgg(input_shape,conv2d_filters,kernel_sizes,initialization,layers,dense_filter1,dense_filter2,dense_filter3,dropout1,dropout2,dropout3,use_bn,initializer,init_dense)
    print(file_name)
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    if opt=='adam':
        optimizer = Adam(lr=lr, decay=init_decay)
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

    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    model = Model([input_a, input_b], distance)
    if loss == "contrastive":
        model.compile(loss=contrastive_loss_altered, optimizer=optimizer, metrics=["accuracy"])
    else:
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    model.summary()

    es = EarlyStopping(monitor='val_acc', patience=es_patience,verbose=1)
    checkpointer = ModelCheckpoint(filepath=weight_file, verbose=2, save_best_only=True)

    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),
                               cooldown=0, patience=lr_patience, min_lr=0.5e-6,verbose=1)

    model.fit([data[0], data[1]], data[2],
          batch_size=batch_size,
          epochs=epochs,
          validation_data=([data[3], data[4]], data[5]),
          callbacks=[es,lr_reducer],
          verbose=2)
    #model = load_model(weight_file) #This is the best model

    score, acc = model.evaluate([data[6], data[7]], data[8], verbose=0)
    print("Test accuracy:%0.3f"% acc)
    pred = model.predict([data[6], data[7]])
    auc_score = roc_auc_score(data[8],pred)
    auc_score = np.round(auc_score,3)
    acc_score = np.round(acc,3)
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
    return acc_score,auc_score

def process_fit(config):
    data = process_data()
    trials = 10
    #conv2d_filters,kernel_sizes,layers,dense_filter1,dense_filter2,dense_filter3,dropout1,dropout2,dropout3,use_bn,batch_size,optimizer,lr,epochs,loss,initializer,init_dense
    conv2d_filters = []
    temp_conv2d_filters = config[0].split("-")
    for i in temp_conv2d_filters:
        conv2d_filters.append(int(i))
    kernel_sizes=int(config[1])
    initialization=config[2]
    layers=config[3] #String Two or Three or anything else
    dense_filter1=int(config[4])
    dense_filter2=int(config[5])
    dense_filter3=int(config[6])
    dropout1=float(config[7])
    dropout2=float(config[8])
    dropout3=float(config[9])
    if config[10]=='FALSE':
        use_bn=False
    else:
        use_bn=True
    batch_size=int(config[11])
    optimizer=config[12]    
    lr=float(config[13])
    epochs=int(config[14])
    loss = config[15]
    initializer = config[16]
    init_dense = config[17]
    accs = []
    aucs = []
    for i in range(trials):
        acc,auc = fit_model(data,conv2d_filters,kernel_sizes,initialization,layers,dense_filter1,dense_filter2,dense_filter3,dropout1,dropout2,dropout3,use_bn,batch_size,optimizer,lr,epochs,loss,initializer,init_dense)
        accs.append(acc)
        aucs.append(auc)
    print("accuracies: ", accs)
    print("aucs: ",aucs)
    mean = np.round(np.mean(aucs),3)
    std = np.round(np.std(aucs),3)
    result = str(mean)+'+/-'+str(std)
    max = np.max(aucs)
    print("mean and std AUC: ",result," max:  ",max)
    return result,max

if __name__ == '__main__':
    search_space = []
    #accuracies = []
    auc_scores = []
    max_aucs = []
    with open('./search_spaces/dn_contrastive_loss_19Nov.csv') as csv_file:
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


