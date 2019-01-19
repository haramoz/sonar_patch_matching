import numpy as np
import random
import h5py
import tensorflow as tf
import pickle
import csv
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import model_from_json
from keras.losses import binary_crossentropy
from keras.optimizers import RMSprop,Adam

from matplotlib.backends.backend_pdf import PdfPages

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics

plt.switch_backend('agg')

def plot_auc(fpr,tpr,filename,auc_score):
    fig=plt.figure()
    plt.plot(fpr, tpr,'m-')
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.3f)' % auc_score)
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

def process_data():
    f = h5py.File('matchedImagesSplitClasses-2017-02-24-17-39-44-96-96-split-val0.15-tr0.7-tst0.15.hdf5','r')
    X_train = f['X_train'].value
    y_train = f['y_train'].value
    X_val = f['X_val'].value
    y_val = f['y_val'].value

    return X_train,y_train,X_val,y_val

def process_data_siamese():
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

def load_model(file_name):
    # load json and create model
    json_file = open(file_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(file_name+".h5")
    print("Loaded model from disk")
    return loaded_model

is_siamese = True
#file_name = 'keras_densenet_siamese_4Nov_15400.9497'
file_name = 'keras_densenet_twochannel_9Nov_14400.9725'

if not is_siamese:
    X_train,y_train,X_val,y_val = process_data()    
    loaded_model = load_model(file_name)
    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    score = loaded_model.evaluate(X_val,y_val, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

    pred = loaded_model.predict(X_val)
    auc_score = roc_auc_score(y_val,pred)
    print("auc_score ------------------> ",np.round(auc_score,3))

    fpr, tpr, thresholds = metrics.roc_curve(y_val, pred)
    plot_auc(fpr,tpr,file_name,auc_score)
else:
    x_train1, x_train2, y_train, x_val1, x_val2, y_val, x_test1, x_test2, y_test = process_data_siamese()
    loaded_model = load_model(file_name)
    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    score = loaded_model.evaluate([x_test1, x_test2],y_test, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

    pred = loaded_model.predict([x_test1, x_test2])
    auc_score = roc_auc_score(y_test,pred)
    print("auc_score ------------------> ",np.round(auc_score,3))

    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)
    #print(fpr,tpr)
    plot_auc(fpr,tpr,file_name,auc_score) 

    """with open('auc_plot.csv', mode='w') as auc_plot:
        auc_plot_writer = csv.writer(auc_plot, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        auc_plot_writer.writerow(fpr)
        auc_plot_writer.writerow(tpr)"""

	

