from __future__ import absolute_import, division, print_function

import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds

def scale_and_crop(max_size, data):
    # data.shape
    # if height is the smalles
    return_data = None
    if data.shape[0] < data.shape[1]:
        scale = max_size / data.shape[0]
        return_data = cv2.resize(data, (int(scale * data.shape[1]), max_size))
        offset = int((return_data.shape[1]-max_size)/2)
        if return_data.shape[1]%2 != 0:
            if offset == 0:
                return_data = return_data[:max_size,:max_size]
            else:
                return_data = return_data[:,offset:offset+max_size]
        else:
            if offset == 0:
                return_data = return_data[:max_size,:max_size]
            else:
                return_data = return_data[:,offset:offset+max_size]

    elif data.shape[0] > data.shape[1]:
        scale = max_size / data.shape[1]
        return_data = cv2.resize(data, (max_size, int(scale * data.shape[0])))
        offset = int((return_data.shape[0]-max_size)/2)
        if return_data.shape[0]%2 != 0:
            if offset == 0:
                return_data = return_data[:max_size,:max_size]
            else:
                return_data = return_data[offset:offset+max_size,:]
        else:
            if offset == 0:
                return_data = return_data[:max_size,:max_size]
            else:
                return_data = return_data[offset:offset+max_size,:]
    else:
        scale = max_size / data.shape[1]
        return_data = cv2.resize(data, dsize=None, fx=scale, fy=scale)

    return return_data

def sanity_no_zero_dimention(data):
    return np.min(data.shape) != 0

def data_prep(data):
    print(tfds.as_numpy(data['image']))
    print(data['file_name'])
    print(data['label'])
    # data_np = tfds.as_numpy(data_in)
    # img_new = scale_and_crop(227, data_np['image'])

    # data = img_new

    # # prepare label
    # label = data_in['label']

    print('In prep functions')
    return data
    # print(label)
    # print(file_name)
    # print(image)
    # return data, label
    

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.ERROR)
    # Load data
    data_loader, info = tfds.load(name='oxford_flowers102', data_dir='/app/data/', with_info=True)
    data_train, data_validation, data_test = data_loader["train"], data_loader["validation"], data_loader['test']

    # print(data_train)

    data_train.map(map_func=data_prep)
    # data_train.get_next()
    # print(info)

    # test_data = cv2.imread('data/image_00001.jpg')[:500,:501]
    # print(test_data.shape)
    # cv2.imwrite('data/testLoad.jpg', test_data)
    # new_test_data = scale_and_crop(227, test_data)
    # cv2.imwrite('data/test_crop.jpg', new_test_data)
    # print(new_test_data.shape)
    # print(cv2.__version__)