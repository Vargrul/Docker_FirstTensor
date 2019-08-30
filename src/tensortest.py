from __future__ import absolute_import, division, print_function

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

# mnist = tf.keras.datasets.mnist

x_train, y_train = mnist_reader.load_mnist('~/data/fashion-mnist', kind='train')
x_test, y_test = mnist_reader.load_mnist('~/data/fashion-mnist', kind='t10k')
x_train, x_test = x_train / 255.0, x_test / 255.0

print(x_train.dtype)
print(y_train.dtype)

model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, verbose=True)
model.evaluate(x_test, y_test)