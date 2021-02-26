#%%
import numpy as np 
import tensorflow as tf 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import h5py
#######################################################################################################

def load_dataset():
    train_dataset = h5py.File('train_happy.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('test_happy.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

# %%
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
# %%


## defining a simple model in keras 
def model(input_shape): 
    X_input = tf.keras.Input(input_shape) 

    # zero padding 
    X = tf.keras.layers.ZeroPadding2D((3,3))(X_input) 

    # CONV -> BN -> RELU Block applied to X
    X = tf.keras.layers.Conv2D(filters = 32, kernel_size= (7,7), name = 'conv0')(X) 
    X = tf.keras.layers.BatchNormalization(axis = 3, name = 'bn0')(X) 
    X = tf.keras.layers.Activation('relu')(X)   

    # MAXPOOL 
    X = tf.keras.layers.MaxPool2D(name = 'max_pool')(X) 

    X = tf.keras.layers.Flatten()(X) 

    X = tf.keras.layers.Dense(1, activation='sigmoid', name = 'fc')(X) 

    model = tf.keras.Model(inputs = X_input, output = X, name = 'HappyModel') 

    return model 



