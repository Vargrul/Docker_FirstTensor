from __future__ import absolute_import, division, print_function

# Add for vscode debugging
# import ptvsd
# ptvsd.enable_attach(address=('0.0.0.0', 7102))
# ptvsd.wait_for_attach()

# TensorFlow and tf.keras
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_datasets as tfds
import data_prep_helper

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Set logging to not see warnings about data loading...
tf.logging.set_verbosity(tf.logging.ERROR)

# Define splits
train_split = tfds.Split.ALL.subsplit(tfds.percent[:90])
# validation_split = tfds.Split.ALL.subsplit(tfds.percent[80:90])
test_split = tfds.Split.ALL.subsplit(tfds.percent[-10:])

# Load data
data_test = tfds.load(name='cats_vs_dogs', data_dir='~/projects/datasets/', split=train_split)
# data_validation = tfds.load(name='cats_vs_dogs', data_dir='~/projects/datasets/', split=validation_split)
data_train = tfds.load(name='cats_vs_dogs', data_dir='~/projects/datasets/', split=test_split)

# For cats_vs_dogs
# data_test, data_validation, data_train = data_loader["train[:80%]"], data_loader["train[80%:90%]"], data_loader['train[90%:]']
# For OxfordFlowers101
# data_test, data_validation, data_train = data_loader["train"], data_loader["validation"], data_loader['test']

data_train = data_train.map(data_prep_helper.data_prep)
# data_validation = data_validation.map(data_prep_helper.data_prep)
data_test = data_test.map(data_prep_helper.data_prep)

data_train = data_train.shuffle(True).batch(128).prefetch(tf.data.experimental.AUTOTUNE)
# data_validation = data_validation.shuffle(True).batch(128).prefetch(tf.data.experimental.AUTOTUNE)
data_test = data_test.shuffle(True).batch(128).prefetch(tf.data.experimental.AUTOTUNE)

# data_train = data_train.make_one_shot_iterator()
# data_validation = data_validation.make_one_shot_iterator()
# data_test = data_test.make_one_shot_iterator()

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
model.add(tf.keras.layers.Dense(2, activation='softmax'))

model.summary()
model.compile(optimizer=tf.train.GradientDescentOptimizer(10e-3), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['mse','accuracy'])

baseline_history = model.fit(data_train, epochs=10)
print(baseline_history)
model.evaluate(data_test)

# Create count of the number of epochs
epoch_count = range(1, len(baseline_history.history['loss']) + 1)

# Visualize loss history
plt.plot(epoch_count, baseline_history.history['loss'])
plt.plot(epoch_count, baseline_history.history['val_loss'])
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig("/app/data/loss.png")
plt.close()

plt.plot(epoch_count, baseline_history.history['acc'])
plt.plot(epoch_count, baseline_history.history['val_acc'])
plt.legend(['Training Acc', 'Test Acc'])
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.savefig("/app/data/acc.png")
# plt.show();
