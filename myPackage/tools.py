import os
from os.path import isfile, join, basename, splitext
from os import listdir
import numpy as np
from numba import jit
from keras.preprocessing import image
from tensorflow.contrib.keras import applications
from matplotlib import pyplot as plt
import itertools
from natsort import natsorted, ns


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
    dict_categories = {}
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
        dict_categories[c] = i
    # print(np.array(X).shape)
    X = applications.inception_v3.preprocess_input(np.array(X))
    # print(X.shape)
    y = np.array(y)
    #n_categories += 1

    return X, y, dict_categories


@jit
def prepateTestDataEval(data_path, dict_categories):
    '''
    Prepare train data from a given path. Extracts the images and the labels
    :param data_path: path where the folders of the images are stored
    :return: array of images and array of labels
    '''
    print("\nPreparing TEST data...\n")
    X = []
    y = []

    for f in os.listdir(os.path.join(data_path, 'test')):
        # print(i, splitext(splitext(basename(f))[0])[0])
        curr = splitext(splitext(basename(f))[0])[0]
        # print(curr)
        img_path = os.path.join(data_path, 'test', f)
        # print(img_path)
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        X += [x]
        y.append(dict_categories[curr])

    # print(np.array(X).shape)
    X = applications.inception_v3.preprocess_input(np.array(X))
    # print(X.shape)
    y = np.array(y)
    # print(y[:4])

    return X, y

@jit
def prepateTestDataPredict(data_path, dict_categories):
    '''
    Prepare train data from a given path. Extracts the images and the labels
    :param data_path: path where the folders of the images are stored
    :return: array of images and array of labels
    '''
    print("\nPreparing TEST data...\n")
    X = []
    y = []

    for f in os.listdir(os.path.join(data_path, 'test')):
        # print(i, splitext(splitext(basename(f))[0])[0])
        curr = splitext(splitext(basename(f))[0])[0]
        # print(curr)
        img_path = os.path.join(data_path, 'test', f)
        # print(img_path)
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        X += [x]
        y.append(dict_categories[curr])

    # print(np.array(X).shape)
    X = applications.inception_v3.preprocess_input(np.array(X))
    # X = np.array(X)
    # print(X.shape)
    y = np.array(y)
    # print(y[:4])

    return X, y


@jit
def split_train_val(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    val_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    train_set = data.iloc[train_indices]
    val_set  = data.iloc[val_indices]
    return train_set.reset_index(drop=True), val_set.reset_index(drop=True)


def getSamples(path, ext=''):
    '''
    Auxiliary function that extracts file names from a given path based on extension
    :param path: source path
    :param ext: file extension
    :return: array with samples
    '''
    samples = [join(path, f) for f in listdir(path)
              if isfile(join(path, f)) and f.endswith(ext)]

    if len(samples) == 0:
        print("ERROR!!! ARRAY OF SAMPLES IS EMPTY (check file extension)")

    return samples


def natSort(list):
    '''
    Sort frames with human method
    see: https://pypi.python.org/pypi/natsort
    :param list: list that will be sorted
    :return: sorted list
    '''
    return natsorted(list, alg=ns.IGNORECASE)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    :param cm: confusion matrix
    :param classes: array of classes' names
    :param normalize: boolean
    :param title: plot title
    :param cmap: colour of matrix background
    :return: plot confusion matrix
    '''

    # plt_name = altsep.join((plot_path,"".join((title,".png"))))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    print("Sum of the main diagonal: {}\n".format(np.trace(cm)))

    # plt.figure()
    # ticks = np.linspace(0, len(cm)-1, num=len(cm))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    plt.title(title)
    # plt.xticks(ticks, fontsize=6)
    # plt.yticks(ticks, fontsize=6)
    # plt.grid(True)
    # plt.show()

    # plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    # plt.colorbar()
    # # tick_marks = np.arange(len(classes))
    # tick_marks = np.linspace(0, len(classes)-1, num= len(classes))
    # plt.xticks(tick_marks)
    # plt.yticks(tick_marks)

    # fmt = '.2f' if normalize else 'd'
    # thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, format(cm[i, j], fmt),
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label', labelpad=0)

    # plt.savefig(plt_name)
    plt.show()