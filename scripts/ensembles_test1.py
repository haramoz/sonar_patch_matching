import numpy as np
import collections
import random
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import argparse

import keras.backend as K
from keras.optimizers import RMSprop,Adam, Adadelta, Nadam
from keras.utils import np_utils
from keras.models import Model,Sequential
from keras.models import model_from_json
import keras.initializers
from keras.losses import binary_crossentropy
#from keras.callbacks import EarlyStopping,ModelCheckpoint

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn import metrics
from matplotlib.backends.backend_pdf import PdfPages
import pickle

seed = 7
np.random.seed(seed)

plt.switch_backend('agg')

def plot_auc(fpr,tpr,filename,auc_score):
    fig=plt.figure()
    #plt.plot(fpr, tpr,'m-')
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',lw=lw, label='DS ROC curve (area = %0.3f)' % auc_score[0])
    plt.plot(fpr[1], tpr[1], color='red',lw=lw, label='DTC ROC curve (area = %0.3f)' % auc_score[1])
    plt.plot(fpr[2], tpr[2], color='green',lw=lw, label='CL ROC curve (area = %0.3f)' % auc_score[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend()
    plt.savefig("figure.png")
    pp = PdfPages(filename+'.pdf')
    pp.savefig(fig)
    pp.close()

def contrastive_loss_altered(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1.0
    return K.mean((1-y_true) * K.square(y_pred) +
                  y_true * K.square(K.maximum(margin - y_pred, 0)))


def process_single_channel_data():
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

    return x_train1, x_train2, y_train_inverted, x_val1, x_val2, y_val_inverted, x_test1, x_test2, y_test

def make_sudo_class_labels(labels):
    indx = len(labels)
    classes = np.asarray(labels,dtype=int)
    counter = 0
    #print(labels[1000:1020])
    for i in range(indx):
        #print(counter)
        if counter >= 10 and counter < 15:
            #print("assigning 2")
            classes[i] = 2 #nonmatching object
        elif counter >= 15 and counter < 20:
            #print("assigning 3")
            classes[i] = 3 #nonmatching background
        elif counter == 20:
            counter = 0
        counter = counter + 1

    #print(classes[1000:1020])
    return classes    


def process_data():
    f = h5py.File('/home/amalli2s/thesis/keras/matchedImagesSplitClasses-2017-02-24-17-39-44-96-96-split-val0.15-tr0.7-tst0.15.hdf5','r')
    X_train = f['X_train'].value
    y_train = f['y_train'].value
    X_test = f['X_val'].value
    y_test = f['y_val'].value
    #X_train_, X_val, y_train_, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)
    
    return X_train,y_train,X_test,y_test

def return_loaded_model(file_name):
    json_file = open(file_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(file_name+".h5")
    print("Loaded model from disk: ",file_name)
    return loaded_model    


if __name__ == '__main__':
    INDX_START = 1000
    INDX_STOP = 1020

    X_train,y_train,X_test,y_test = process_data() 
    _,_,_,_,_,_,x_test1, x_test2,_= process_single_channel_data()
    file_name = 'final_auc_compare'    
    #class_label = make_sudo_class_labels(y_test)
    # load json and create model
    #file_name_dtc = 'keras_densenet_twochannel_7Nov_14400.9593'
    file_name_dtc = 'keras_densenet_twochannel_14Dec_00000.9664' #best config
    #file_name_dtc = 'keras_densenet_twochannel_9Nov_14400.9725'
    #file_name_dtc = 'keras_densenet_twochannel_14Dec_00000.9632'
    #file_name_dtc = 'keras_densenet_twochannel_14Dec_00000.9704'
    #file_name_ds = 'keras_densenet_siamese_4Nov_15400.9524'
    file_name_ds = 'keras_densenet_siamese_4Nov_15400.9497'
    #file_name_ds = 'keras_densenet_siamese_27Oct_04200.9491'

    file_name_cl = 'keras_contrastive_loss_20Dec2pm0.956' #best config


    #load densenet two channel model
    model_dtc = return_loaded_model(file_name_dtc)

    #load contrastive loss model
    model_cl = return_loaded_model(file_name_cl)

    #load Densenet siamese model
    model_ds = return_loaded_model(file_name_ds)

    optimizer_dtc = Adadelta(lr=0.03)
    optimizer_cl = Nadam(lr=0.0002)
    optimizer_ds = Adadelta(lr=0.07)
    model_dtc.compile(loss='binary_crossentropy', optimizer=optimizer_dtc, metrics=['accuracy'])
    model_ds.compile(loss='binary_crossentropy', optimizer=optimizer_ds, metrics=['accuracy'])
    model_cl.compile(loss=contrastive_loss_altered, optimizer=optimizer_cl, metrics=['accuracy'])

    prediction_dtc = model_dtc.predict(X_test)
    prediction_ds = model_ds.predict([x_test1,x_test2])
    prediction_cl = model_cl.predict([x_test1,x_test2])
    aucs = []
    auc_score = roc_auc_score(y_test, prediction_ds)
    aucs.append(auc_score)
    auc_score = roc_auc_score(y_test, prediction_dtc)
    aucs.append(auc_score)
    auc_score = roc_auc_score((1-y_test), prediction_cl)
    aucs.append(auc_score)
    #print("prediction dtc \n",prediction_dtc[INDX_START:INDX_STOP])
    #print("prediction ds \n",prediction_ds[INDX_START:INDX_STOP])
    #print("prediction cl \n",prediction_cl[INDX_START:INDX_STOP])
    #print("ground truth ",y_test[INDX_START:INDX_STOP])
    fpr = []
    tpr = []
    f, t,_ = metrics.roc_curve(y_test, prediction_ds)
    fpr.append(f)
    tpr.append(t)
    f, t,_ = metrics.roc_curve(y_test, prediction_dtc)
    fpr.append(f)
    tpr.append(t)
    f, t,_ = metrics.roc_curve((1-y_test), prediction_cl)
    fpr.append(f)
    tpr.append(t)
    plot_auc(fpr,tpr,file_name,aucs)



    """with open('all_prediction_analysis.csv', mode='w') as ds_file:
        ds_writer = csv.writer(ds_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for i in range(len(prediction_ds)):
            ds_writer.writerow([i,prediction_ds[i],prediction_dtc[i],prediction_cl[i],y_test[i],1-y_test[i]])
        ds_file.close()"""

