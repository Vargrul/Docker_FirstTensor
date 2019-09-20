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

import time

# Setting up timing

print(tf.version)

# data_path = '/user/create.aau.dk/ksla/data' #compute data location
data_path = '/home/ksla/projects/datasets' #local data location

x_test, y_test = mnist_reader.load_mnist(data_path + '/fashion-mnist', kind='t10k')
x_train, y_train = mnist_reader.load_mnist(data_path + '/fashion-mnist', kind='train')
x_train, x_test = x_train / 255.0, x_test / 255.0

data_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
data_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# data_train = data_train.map(data_prep_helper.data_prep)
# data_test = data_test.map(data_prep_helper.data_prep)

data_train = data_train.shuffle(True).batch(512).prefetch(tf.data.experimental.AUTOTUNE)
data_test = data_test.shuffle(True).batch(512).prefetch(tf.data.experimental.AUTOTUNE)

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

t0 = time.time()
model_1.fit(data_train, epochs=25, verbose=True)
model_1_time = time.time() - t0

t0 = time.time()
model_2.fit(data_train, epochs=25, verbose=True)
model_2_time = time.time() - t0

t0 = time.time()
model_3.fit(data_train, epochs=25, verbose=True)
model_3_time = time.time() - t0

t0 = time.time()
model_4.fit(data_train, epochs=25, verbose=True)
model_4_time = time.time() - t0

model_1.evaluate(data_test)
model_2.evaluate(data_test)
model_3.evaluate(data_test)
model_4.evaluate(data_test)

print(model_1_time)
print(model_2_time)
print(model_3_time)
print(model_4_time)