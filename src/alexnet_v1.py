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
data_train, data_validation, data_test = data_loader["train"], data_loader["validation"], data_loader['test']

# data_train = data_train.shuffle(1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
# data_validation = data_validation.shuffle(1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
# data_test = data_test.shuffle(1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE)


# print(data_train.output_types)
# print(data_train.output_shapes)

print(data_test)

data_test = data_test.map(data_prep_helper.data_prep)
data_validation = data_validation.map(data_prep_helper.data_prep)

print(data_test)

data_test = data_test.shuffle(True).batch(32).repeat(5)
data_validation = data_validation.shuffle(True).batch(32).repeat(5)
print(data_test)

print('convertin to iterator')
data_test = data_test.make_one_shot_iterator()
data_validation = data_validation.make_one_shot_iterator()

print(data_test)

# data_train_np_scaled = np.zeros((info.splits['train'].num_examples,227,227,3), dtype=np.uint8)
# label_train = np.zeros(info.splits['train'].num_examples, dtype=np.int32)
# data_validation_np_scaled = np.zeros((info.splits['validation'].num_examples,227,227,3), dtype=np.uint8)
# label_validation = np.zeros(info.splits['validation'].num_examples, dtype=np.int32)
# # data_test_np_scaled = np.zeros((info.splits['test'].num_examples,227,227,3), dtype=np.uint8)
# # label_test = np.zeros(info.splits['test'].num_examples, dtype=np.int32)

# print(data_train_np_scaled.shape)
# print(label_train.shape)
# print(data_validation_np_scaled.shape)
# print(label_validation.shape)
# # print(data_test_np_scaled.shape)
# # print(label_test.shape)

# for idx, i in enumerate(tfds.as_numpy(data_train)):

#     # prepare image
#     img_new = data_prep_helper.scale_and_crop(227, i['image'])
#     if not data_prep_helper.sanity_no_zero_dimention(img_new):
#         print(idx, end='')
#         print('\t', end='')
#         print(i['image'].shape)
#     data_train_np_scaled[idx] = img_new

#     # prepare label
#     label_train[idx] = i['label']
# data_train_np_scaled = data_train_np_scaled / 255.0
    
# for idx, i in enumerate(tfds.as_numpy(data_validation)):

#     # prepare image
#     img_new = data_prep_helper.scale_and_crop(227, i['image'])
#     if not data_prep_helper.sanity_no_zero_dimention(img_new):
#         print(idx, end='')
#         print('\t', end='')
#         print(i['image'].shape)
#     data_validation_np_scaled[idx] = img_new
    

#     # print(img_new.shape)

#     # prepare label
#     label_validation[idx] = i['label']
# data_validation_np_scaled = data_validation_np_scaled / 255.0

# for idx, i in enumerate(tfds.as_numpy(data_test)):

#     # prepare image
#     img_new = data_prep_helper.scale_and_crop(227, i['image'])
#     if not data_prep_helper.sanity_no_zero_dimention(img_new):
#         print(idx, end='')
#         print('\t', end='')
#         print(i['image'].shape)
#     data_test_np_scaled[idx] = img_new

#     # print(img_new.shape)

#     # prepare label
#     label_test = i['label']
# data_test_np_scaled = data_test_np_scaled / 255.0
    
# data_exmaple = data_train.take(1)
# # print(data_exmaple[1])
# label = data_exmaple['label']
# print(label.numpy())
# , info_validate = tfds.load(name='oxford_flowers102', split=tfds.Split.VALIDATION, data_dir='/app/data/', with_info=True)

# print(data_train)

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

model.add(tf.keras.layers.Dense(4096, 'relu'))
model.add(tf.keras.layers.Dense(4096, 'relu'))
model.add(tf.keras.layers.Dense(102, activation='softmax'))

model.summary()
model.compile(optimizer=tf.train.GradientDescentOptimizer(0.01), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

# print(model.predict(data_train_np_scaled[0:1]))
baseline_history = model.fit(data_test, epochs=5, batch_size=32, steps_per_epoch=30, validation_data=data_validation, validation_steps=30)
# baseline_history = model.fit(data_train_np_scaled, label_train, epochs=5, verbose=1, batch_size=8, validation_data=(data_validation_np_scaled, label_validation))
# model.evaluate()
