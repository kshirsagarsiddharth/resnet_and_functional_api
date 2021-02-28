# lets create a toy resnet model 
#######################################################################3
## 
#%%
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
outputs = layers.Dense(10)(x)

model = tf.keras.Model(inputs, outputs, name = 'totcifardataet')
tf.keras.utils.plot_model(model)



tf.keras.utils.plot_model(model)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255.0 
x_test = x_test.astype("float32") / 255.0 

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)



model.compile(
optimizer = tf.keras.optimizers.RMSprop(1e-3),
    loss = tf.keras.losses.CategoricalCrossentropy(), 
    metrics = ['acc']

)


model.fit(x_train[:2000], y_train[:2000], batch_size = 128, epochs = 4,
         validation_split = 0.2
         
         )







# %%
