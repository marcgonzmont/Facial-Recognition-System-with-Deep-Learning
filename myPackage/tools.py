import os
from os.path import basename, splitext
import numpy as np
from numba import jit
from keras.preprocessing import image
from tensorflow.contrib.keras import applications

@jit
def prepareTrainData(data_path):
    '''
    Prepare train data from a given path. Extracts the images and the labels
    :param data_path: path where the folders of the images are stored
    :return: array of images and array of labels
    '''
    print("\nPreparing TRAIN data...")
    X = []
    y = []
    n_categories = 0
    categories = os.path.join(data_path, 'train')

    for i, c in enumerate(os.listdir(categories)):
        for f in os.listdir(os.path.join(data_path, 'train', c)):
            # print(i, c, f)
            img_path = os.path.join(data_path, 'train', c, f)
            # print(img_path)
            img = image.load_img(img_path, target_size=(299, 299))
            x = image.img_to_array(img)
            X += [x]
            y += [i]
        n_categories = i

    X = applications.inception_v3.preprocess_input(np.array(X))
    y = np.array(y)
    n_categories += 1

    return X, y, n_categories


@jit
def prepateTestData(data_path):
    '''
    Prepare train data from a given path. Extracts the images and the labels
    :param data_path: path where the folders of the images are stored
    :return: array of images and array of labels
    '''
    print("\nPreparing TEST data...")
    X = []
    y = []
    i = 0
    prev = ''

    for f in os.listdir(os.path.join(data_path, 'test')):
        # print(i, splitext(splitext(basename(f))[0])[0])
        curr = splitext(splitext(basename(f))[0])[0]
        img_path = os.path.join(data_path, 'test', f)
        # print(img_path)
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        X += [x]
        y += [i]
        if prev == curr:
            i += 1
        else:
            prev = curr

    X = applications.inception_v3.preprocess_input(np.array(X))
    y = np.array(y)

    return X, y


def split_train_val(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    val_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    train_set = data.iloc[train_indices]
    val_set  = data.iloc[val_indices]
    return train_set.reset_index(drop=True), val_set.reset_index(drop=True)

@jit
def getSamples(path):
    print("path ", path)
    samples = [os.path.join(path, f) for f in os.listdir(path)]
    return samples