# lets create a toy resnet model 
#######################################################################3
## 

import tensorflow as tf
from tensorflow.keras import layers 

inputs = tf.keras.Input(shape = (32,32,3), name = 'img')
x = layers.Conv2D(32,3, activation = 'relu')(inputs)
x = layers.Conv2D(64,3, activation = 'relu')(x)
block_1_output = layers.MaxPool2D(3)(x)

x = layers.Conv2D(64,3, activation = 'relu', padding = 'same')(block_1_output)
x = layers.Conv2D(64,3, activation = 'relu', padding = 'same')(x)

block_2_output = layers.add([x, block_1_output])

x = layers.Conv2D(64,3, activation = 'relu', padding = 'same')(block_2_output)
x = layers.Conv2D(64,3, activation = 'relu', padding = 'same')(x)

block_3_output = layers.add([x, block_2_output])

x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_3_output)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(10)(x)










