from __future__ import absolute_import, division, print_function

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Add for vscode debugging
# import ptvsd
# ptvsd.enable_attach(address=('0.0.0.0', 7102))
# ptvsd.wait_for_attach()

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import mnist_reader
import models_mnist

# mnist = tf.keras.datasets.mnist

print(tf.version)

x_test, y_test = mnist_reader.load_mnist('/home/ksla/projects/datasets/fashion-mnist', kind='t10k')
x_train, y_train = mnist_reader.load_mnist('/home/ksla/projects/datasets/fashion-mnist', kind='train')
x_train, x_test = x_train / 255.0, x_test / 255.0

data_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
data_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))


print(type(data_train))
print(type(data_test))

# data_train = data_train.map(data_prep_helper.data_prep)
# data_test = data_test.map(data_prep_helper.data_prep)

data_train = data_train.shuffle(True).batch(512).prefetch(tf.data.experimental.AUTOTUNE)
data_test = data_test.shuffle(True).batch(512).prefetch(tf.data.experimental.AUTOTUNE)

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
#     tf.keras.layers.Dense(128, activation=tf.nn.relu),
#     tf.keras.layers.Dense(10, activation=tf.nn.relu),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(10, activation=tf.nn.softmax)
# ])
model_1 = models_mnist.model_simple_dense()
model_2 = models_mnist.model_1_conv()
model_3 = models_mnist.model_3_conv()
model_4 = models_mnist.model_4_conv()

model_1.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model_2.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model_3.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model_4.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])                                          

model_1.fit(data_train, epochs=25, verbose=True)
model_2.fit(data_train, epochs=25, verbose=True)
model_3.fit(data_train, epochs=25, verbose=True)
model_4.fit(data_train, epochs=25, verbose=True)
model_1.evaluate(data_test)
model_2.evaluate(data_test)
model_3.evaluate(data_test)
model_4.evaluate(data_test)
