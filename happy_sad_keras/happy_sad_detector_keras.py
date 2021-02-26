#%%
import numpy as np 
import tensorflow as tf 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import h5py
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import layer_utils
from tensorflow.keras.utils.data_utils import get_file
from tensorflow.keras.applications.imagenet_utils import preprocess_input
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


# GRADED FUNCTION: HappyModel

def HappyModel(input_shape):
    """
    Implementation of the HappyModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset
        (height, width, channels) as a tuple.  
        Note that this does not include the 'batch' as a dimension.
        If you have a batch like 'X_train', 
        then you can provide the input_shape using
        X_train.shape[1:]
    

    Returns:
    model -- a Model() instance in Keras
 """
    
    ### START CODE HERE ###
    # Feel free to use the suggested outline in the text above to get started, and run through the whole
    # exercise (including the later portions of this notebook) once. The come back also try out other
    # network architectures as well. 
    
    X_input = Input(input_shape) 
    
    X = ZeroPadding2D((3,3))(X_input) 
    X = Conv2D(32, (7,7), name = 'conv0')(X) 
    X = BatchNormalization(axis=3, name = "bn0")(X) 
    X = Activation('relu')(X)
    
    X = MaxPooling2D(name = 'max_pool1')(X)
    
    X = ZeroPadding2D((3,3))(X) 
    X = Conv2D(64, (7,7), name = 'conv1')(X) 
    X = BatchNormalization(axis=3, name = "bn1")(X) 
    X = Activation('relu')(X) 
    
    X = MaxPooling2D(name = 'mp2')(X)
    
    X = Flatten()(X) 
    X = Dense(16, activation='relu')(X)
    X = Dropout(0.3)(X)
    X = Dense(1, activation='sigmoid', name = 'fc')(X)

    
    model = Model(inputs = X_input, outputs = X, name = 'HappyModel')
    ### END CODE HERE ###
    
    return model

### START CODE HERE ### (1 line)
happyModel = HappyModel(X_train.shape[1:])
### END CODE HERE ###
### START CODE HERE ### (1 line)
happyModel.compile(loss='binary_crossentropy',optimizer='adam',metrics = ["accuracy"])
### END CODE HERE ###


### START CODE HERE ### (1 line)
happyModel.fit(X_train,Y_train,batch_size = 128,epochs = 300, validation_split=0.1)
### END CODE HERE ###

### START CODE HERE ### (1 line)
preds = happyModel.evaluate(X_test,Y_test)
### END CODE HERE ###
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
# %%
