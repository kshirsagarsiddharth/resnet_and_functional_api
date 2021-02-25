#%%
import numpy as np 
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers 

"""
(input: 784-dimensional vectors)
       ↧
[Dense (64 units, relu activation)]
       ↧
[Dense (64 units, relu activation)]
       ↧
[Dense (10 units, softmax activation)]
       ↧
(output: logits of a probability distribution over 10 classes)
"""

# the goal is to build this network 
#inputs = tf.keras.Input(shape = (784,))
# the shape of data is 784 dimensional vector. 
# the inputs object returns shape and datatype of the tensor 
#print(inputs.shape) 
#print(inputs.dtype)

# creating a new node in the graph of layers by calling a layer on this inputs object 
#dense = layers.Dense(64, activation='relu') 
#x = dense(inputs)
# the "layer call" action is like drawing an arrow from inputs to the "created layer" 
# i.e passing the inputs to the dense layer and we get x as output 
#x  = layers.Dense(64, activation='relu')(x)
#outputs = layers.Dense(10)(x)
# %%
# creating the model
inputs = tf.keras.Input(shape=(784,))
dense1 = layers.Dense(64, activation='relu') (inputs)
dense2 = layers.Dense(64, activation='relu')(dense1) 
outputs = layers.Dense(10)(dense2) 

# creating the model 
model = tf.keras.Model(inputs = inputs, outputs = outputs, name = 'mnist_model')  
model.summary()
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png')
# %%
#import tf.keras
keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)
# %%

# %%
