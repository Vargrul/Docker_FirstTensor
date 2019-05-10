from __future__ import absolute_import, division, print_function

# Add for vscode debugging
import ptvsd
ptvsd.enable_attach(address=('0.0.0.0', 7102))
# ptvsd.wait_for_attach()

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import data_prep_helper

# Helper libraries
import numpy as np

# Set logging to not see warnings about data loading...
tf.logging.set_verbosity(tf.logging.ERROR)

# Load data
data_loader, info = tfds.load(name='oxford_flowers102', data_dir='/app/data/', with_info=True)
data_test, data_validation, data_train = data_loader["train"], data_loader["validation"], data_loader['test']

data_train = data_train.map(data_prep_helper.data_prep)
data_validation = data_validation.map(data_prep_helper.data_prep)
data_test = data_test.map(data_prep_helper.data_prep)

data_train = data_train.shuffle(True).batch(128).prefetch(tf.data.experimental.AUTOTUNE).repeat()
data_validation = data_validation.shuffle(True).batch(128).prefetch(tf.data.experimental.AUTOTUNE).repeat()
data_test = data_test.shuffle(True).batch(128).prefetch(tf.data.experimental.AUTOTUNE)

data_train = data_train.make_one_shot_iterator()
data_validation = data_validation.make_one_shot_iterator()
data_test = data_test.make_one_shot_iterator()

# Make model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(96, (11,11), input_shape=(227,227,3), strides=(4,4), activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D(pool_size=(5,5), strides=(2,2)))

model.add(tf.keras.layers.Conv2D(256, (5,5), activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D())

model.add(tf.keras.layers.Conv2D(384, (3,3), activation='relu'))

model.add(tf.keras.layers.Conv2D(384, (3,3), activation='relu'))

model.add(tf.keras.layers.Conv2D(384, (3,3), activation='relu'))

model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(4096, 'relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(4096, 'relu'))
model.add(tf.keras.layers.Dense(102, activation='softmax'))

model.summary()
model.compile(optimizer=tf.train.GradientDescentOptimizer(10e-3), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['mse','accuracy'])

baseline_history = model.fit(data_train, epochs=15, batch_size=128, steps_per_epoch=48, validation_data=data_validation, validation_steps=8)
model.evaluate(data_test, steps=8)
