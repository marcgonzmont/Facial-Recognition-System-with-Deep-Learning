import os
# from os import altsep
import numpy as np
from keras.preprocessing import image
from tensorflow.contrib.keras import applications

def prepareTrainData(data_path):
    '''
    Prepare train data from a given path. Extracts the images and the labels
    :param data_path: path where the folders of the images are stored
    :return: array of images and array of labels
    '''
    X = []
    y = []
    n_categories = 0
    categories = os.path.join(data_path, 'train')

    for i, c in enumerate(os.listdir(categories)):
        for f in os.listdir(os.path.join(data_path, 'train', c)):
            # print(i, c, f)
            img_path = os.path.join(data_path, 'train', c, f)
            img = image.load_img(img_path, target_size=(299, 299))
            x = image.img_to_array(img)
            X += [x]
            y += [c]
        n_categories = i

    X = applications.inception_v3.preprocess_input(np.array(X))
    y = np.array(y)

    return X, y, n_categories


def prepateTestData(data_path):
    X = []
    y = []

    for f in os.listdir(os.path.join(data_path, 'test')):
        img_path = os.path.join(data_path, 'test', f)
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        X += [x]
        y += [os.path.splitext(os.path.splitext(os.path.basename(img_path))[0])]

    X = applications.inception_v3.preprocess_input(np.array(X))
    y = np.array(y)

    return X, y


def getSamples(path):
    print("path ", path)
    samples = [os.path.join(path, f) for f in os.listdir(path)]
    return samples