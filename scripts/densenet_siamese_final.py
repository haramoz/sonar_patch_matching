"""
https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/DenseNet
"""
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
from keras.layers import Flatten, Conv2D
from keras.regularizers import l2
from keras.optimizers import RMSprop,Adam
from keras.utils import np_utils
from keras.models import Model,Sequential
from keras.wrappers.scikit_learn import KerasClassifier


from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
# seed for reproducing same results

seed = 7
np.random.seed(seed)

f = h5py.File('matchedImagesSplitClasses-2017-02-24-17-39-44-96-96-split-val0.15-tr0.7-tst0.15.hdf5','r')
ln_training = 39840
X_train = f['X_train'].value
X_train_resize = X_train[0:ln_training,:,:,:]
#print(X_train_resize.shape)
y_train = f['y_train'].value
y_train_resize = y_train[0:ln_training,]
y_train_categorical = np_utils.to_categorical(y_train_resize, 2)
#X_reshaped = X_train_resize.reshape(*X_train_resize.shape[:1], -2)
#print(X_reshaped.shape)
ln_validation = 7440
X_val = f['X_val'].value
X_val_resize = X_val[0:ln_validation,:,:,:]

#X_val_reshaped = X_val
#X_val_reshaped = X_val_resize.reshape(*X_val_resize.shape[:1], -2)

y_val = f['y_val'].value
y_val_reshaped = y_val[0:ln_validation]
y_val_categorical = np_utils.to_categorical(y_val_reshaped, 2)
#print(X_val_reshaped.shape)
#print(y_val_categorical.shape)

tr_pair1 = X_train_resize[:,0,:,:]
tr_pair1_reshaped = tr_pair1.reshape(X_train_resize.shape[0],1,X_train_resize.shape[2],X_train_resize.shape[3])
tr_pair2 = X_train_resize[:,1,:,:]
tr_pair2_reshaped = tr_pair2.reshape(X_train_resize.shape[0],1,X_train_resize.shape[2],X_train_resize.shape[3])
print (tr_pair1_reshaped.shape,tr_pair2_reshaped.shape)
print (y_train_resize.shape)

te_pair1 = X_val_resize[:,0,:,:]
te_pair1_reshaped = te_pair1.reshape(X_val_resize.shape[0],1,X_val_resize.shape[2],X_val_resize.shape[3])
te_pair2 = X_val_resize[:,1,:,:]
te_pair2_reshaped = te_pair2.reshape(X_val_resize.shape[0],1,X_val_resize.shape[2],X_val_resize.shape[3])
print (te_pair1_reshaped.shape,te_pair2_reshaped.shape)

#default values
nb_classes = len(np.unique(y_train)) #The number of classes here are matching and nonmatching so only 2
img_dim = (1,96,96) #X_train.shape[2:]
n_channels = 1 #(1,96,96) #X_train.shape[1] #2
batch_size = 32
nb_epochs = 3
depth =  10 #40 #3N+4
nb_filter = 16 #16 #increasing this worsens the result
nb_dense_block = 1 #3
growth_rate = 12
dropout_rate = 0.3
learning_rate = 1E-3
weight_decay = 1E-4
plot_architecture = False

""" The densenet functions from tdeboissiere"""
def conv_factory(x, concat_axis, nb_filter,
                 dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 3x3Conv2D, optional dropout
    :param x: Input keras network
    :param concat_axis: int -- index of contatenate axis
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: keras network with b_norm, relu and Conv2D added
    :rtype: keras network
    """

    x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (3, 3),
               kernel_initializer="he_uniform",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition(x, concat_axis, nb_filter,
               dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 1x1Conv2D, optional dropout and Maxpooling2D
    :param x: keras model
    :param concat_axis: int -- index of contatenate axis
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: model
    :rtype: keras model, after applying batch_norm, relu-conv, dropout, maxpool
    """

    x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (1, 1),
               kernel_initializer="he_uniform",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)

    return x


def denseblock(x, concat_axis, nb_layers, nb_filter, growth_rate,
               dropout_rate=None, weight_decay=1E-4):
    """Build a denseblock where the output of each
       conv_factory is fed to subsequent ones
    :param x: keras model
    :param concat_axis: int -- index of contatenate axis
    :param nb_layers: int -- the number of layers of conv_
                      factory to append to the model.
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model
    """

    list_feat = [x]

    for i in range(nb_layers):
        x = conv_factory(x, concat_axis, growth_rate,
                         dropout_rate, weight_decay)
        list_feat.append(x)
        x = Concatenate(axis=concat_axis)(list_feat)
        nb_filter += growth_rate

    return x, nb_filter


def denseblock_altern(x, concat_axis, nb_layers, nb_filter, growth_rate,
                      dropout_rate=None, weight_decay=1E-4):
    """Build a denseblock where the output of each conv_factory
       is fed to subsequent ones. (Alternative of a above)
    :param x: keras model
    :param concat_axis: int -- index of contatenate axis
    :param nb_layers: int -- the number of layers of conv_
                      factory to append to the model.
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model
    * The main difference between this implementation and the implementation
    above is that the one above
    """

    for i in range(nb_layers):
        merge_tensor = conv_factory(x, concat_axis, growth_rate,
                                    dropout_rate, weight_decay)
        x = Concatenate(axis=concat_axis)([merge_tensor, x])
        nb_filter += growth_rate

    return x, nb_filter


def DenseNet(nb_classes, img_dim, depth, nb_dense_block, growth_rate,
             nb_filter, dropout_rate=None, weight_decay=1E-4):
    """ Build the DenseNet model
    :param nb_classes: int -- number of classes
    :param img_dim: tuple -- (channels, rows, columns)
    :param depth: int -- how many layers
    :param nb_dense_block: int -- number of dense blocks to add to end
    :param growth_rate: int -- number of filters to add
    :param nb_filter: int -- number of filters
    :param dropout_rate: float -- dropout rate
    :param weight_decay: float -- weight decay
    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model
    """
    
    if K.image_dim_ordering() == "th":
        concat_axis = 1
    elif K.image_dim_ordering() == "tf":
        concat_axis = -1

    model_input = Input(shape=img_dim)

    assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"

    # layers in each dense block
    nb_layers = int((depth - 4) / 3)

    # Initial convolution
    x = Conv2D(nb_filter, (3, 3),
               kernel_initializer="he_uniform",
               padding="same",
               name="initial_conv2D",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(model_input)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = denseblock(x, concat_axis, nb_layers,
                                  nb_filter, growth_rate, 
                                  dropout_rate=dropout_rate,
                                  weight_decay=weight_decay)
        # add transition
        x = transition(x, nb_filter, dropout_rate=dropout_rate,
                       weight_decay=weight_decay)

    # The last denseblock does not have a transition
    x, nb_filter = denseblock(x, concat_axis, nb_layers,
                              nb_filter, growth_rate, 
                              dropout_rate=dropout_rate,
                              weight_decay=weight_decay)

    x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    """x = GlobalAveragePooling2D(data_format=K.image_data_format())(x)
    x = Dense(nb_classes,
              activation='softmax',
              kernel_regularizer=l2(weight_decay),
              bias_regularizer=l2(weight_decay))(x)"""
    

    densenet = Model(inputs=[model_input], outputs=[x], name="DenseNet")
    #densenet = Lambda(inputs=[model_input], outputs=[x], name="DenseNet")
    
    return densenet

def create_base_network(img_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''    
    net = DenseNet(nb_classes,
                  img_dim,
                  depth,
                  nb_dense_block,
                  growth_rate,
                  nb_filter,
                  dropout_rate=dropout_rate,
                  weight_decay=weight_decay)
    
    return net

def plot_auc(fpr,tpr):
    plt.figure()
    plt.plot(fpr, tpr,'m-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.show()

###################
# Construct model #
###################
def createModel():
    base_network = create_base_network(img_dim) #think this is correct

    input_a = Input(shape=(img_dim))
    input_b = Input(shape=(img_dim))

    #distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    img_a_feat = base_network(input_a)
    img_b_feat = base_network(input_b)
    print(img_a_feat.shape)
    combined_features = concatenate([img_a_feat, img_b_feat], name = 'merge_features')
    combined_features = Dense(64)(combined_features)
    combined_features = BatchNormalization()(combined_features)
    combined_features = Activation('relu')(combined_features)
    combined_features = Dense(1, activation = 'sigmoid')(combined_features)

    similarity_model = Model(inputs = [input_a, input_b], outputs = [combined_features], name = 'Similarity_Model')

    similarity_model.summary()
    similarity_model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])  
    return similarity_model

#opt = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
created_model = createModel()
created_model.fit([tr_pair1_reshaped, tr_pair2_reshaped],y_train, epochs=3,batch_size=32,verbose=1)

#scores = similarity_model.evaluate([te_pair1_reshaped,te_pair2_reshaped], y_val_reshaped)
#print(scores)
pred = created_model.predict([te_pair1_reshaped,te_pair2_reshaped])
fpr, tpr, thresholds = metrics.roc_curve(y_val_reshaped, pred)
print(fpr,tpr)
roc_auc = metrics.auc(fpr, tpr)
print(roc_auc)
plot_auc(fpr,tpr)
