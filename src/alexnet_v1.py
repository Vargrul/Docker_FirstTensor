from __future__ import absolute_import, division, print_function

# Add for vscode debugging
import ptvsd
ptvsd.enable_attach(address=('0.0.0.0', 7102))
# ptvsd.wait_for_attach()

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds

# Helper libraries
import numpy as np

# Set logging to not see warnings about data loading...
tf.logging.set_verbosity(tf.logging.ERROR)

# Load data
data_train, info_train = tfds.load(name='oxford_flowers102', split=tfds.Split.TRAIN, data_dir='/app/data/', with_info=True)
data_validate, info_validate = tfds.load(name='oxford_flowers102', split=tfds.Split.VALIDATION, data_dir='/app/data/', with_info=True)

print(info_train)
print(info_validate)

# Make model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(96, (11,11), input_shape=(224,224,3), strides=(4,4), activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D(pool_size=(5,5), strides=(2,2)))

model.add(tf.keras.layers.Conv2D(256, (5,5), activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D())

model.add(tf.keras.layers.Conv2D(384, (3,3), activation='relu'))

model.add(tf.keras.layers.Conv2D(384, (3,3), activation='relu'))

model.add(tf.keras.layers.Conv2D(384, (3,3), activation='relu'))

model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu'))

model.add(tf.keras.layers.Dense(4096, 'relu'))
model.add(tf.keras.layers.Dense(4096, 'relu'))
model.add(tf.keras.layers.Dense(102, activation='softmax'))

model.summary()

model.compile(optimizer=tf.train.GradientDescentOptimizer(0.01), loss='categorical_crossentropy', metric=['accuracy', 'precision'])

model.fit(data_train, batch_size=32, epochs=1, verbose=1, validation_data=data_validate)
