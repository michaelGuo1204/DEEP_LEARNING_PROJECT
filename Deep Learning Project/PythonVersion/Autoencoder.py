from random import randint
from numpy import load
from numpy import array
from numpy import argmax
from numpy import array_equal
from Model import Autoencoder_Model,divide_train_test

#%% Load Data
data = load('XY.npz')
X_onehot = data['arr_0']
Y = data['arr_1']
X_train, Y_train, X_val, Y_val, X_test, Y_test = divide_train_test(X_onehot, Y, proportion=(0.8,0.1,0.1))

#%% Set Model
autoencoder=Autoencoder_Model(ifAttention=True)
autoencoder.trainModel(X_train,Y_train,X_val,Y_val)