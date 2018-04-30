from tensorflow.contrib.keras import applications, layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
from numba import jit
from matplotlib import pyplot as plt

@jit
def loadModel(n_classes):
    '''
    This function loads a pre-trained Keras model
    :param n_classes: number of classes to configure the model
    :return: model configured
    '''
    print("\nLoading model...")
    base_model = applications.inception_v3.InceptionV3(input_shape= (299, 299, 3), weights= 'imagenet', include_top= False)
    # add a global spatial average pooling layer
    x = layers.GlobalAveragePooling2D()(base_model.output)

    # Add the prediction layer of size n_classes
    predictions = layers.Dense(n_classes, activation= 'softmax')(x)
    model = models.Model(inputs= base_model.input, outputs= predictions)

    for layer in base_model.layers:
        layer.trainable = False

    opt = optimizers.RMSprop(lr= 0.001)
    model.compile(loss= 'sparse_categorical_crossentropy', optimizer= opt, metrics= ['accuracy']) # add precision, recall

    return model


@jit
def trainGenModel(model, batch_size, n_epoch, X, y, val_split= None):
    '''
    Train the loaded model
    :param model: model loaded
    :param X: training instances
    :param y: training labels
    :return: trained model
    '''
    print("\nTraining model with data augmentation...")
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')
    # generated = datagen.fit(X)
    if val_split != None:
        history = model.fit_generator(datagen.flow(X, y, batch_size= batch_size),
                    steps_per_epoch=len(X) / batch_size, epochs= n_epoch, use_multiprocessing= True)

        #  "Accuracy"
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()
        # "Loss"
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()
    else:
        history = model.fit_generator(datagen.flow(X, y, batch_size=batch_size),
                                      steps_per_epoch=len(X) / batch_size, epochs=n_epoch, workers=4)
        #  "Accuracy"
        plt.plot(history.history['acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train'], loc='upper left')
        plt.show()
        # "Loss"
        plt.plot(history.history['loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train'], loc='upper left')
        plt.show()
    # if val_split != 0.0:
    #     plt.plot(history.history['val_acc'])
    #     plt.show()

    return model

@jit
def trainModel(model, batch_size, n_epoch, X, y, val_split= None):
    '''
    Train the loaded model
    :param model: model loaded
    :param X: training instances
    :param y: training labels
    :return: trained model
    '''
    print("\nTraining model...")
    if val_split != None:
        history = model.fit(x= X, y= y, batch_size= batch_size, epochs= n_epoch, verbose= 1, validation_split= val_split)

        #  "Accuracy"
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()
        # "Loss"
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()
    else:
        history = model.fit(x=X, y=y, batch_size=batch_size, epochs=n_epoch, verbose=1)
        #  "Accuracy"
        plt.plot(history.history['acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train'], loc='upper left')
        plt.show()
        # "Loss"
        plt.plot(history.history['loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train'], loc='upper left')
        plt.show()
    # if val_split != 0.0:
    #     plt.plot(history.history['val_acc'])
    #     plt.show()

    return model


@jit
def testModel(model, batch_size, X, y):
    '''
    Test the model
    :param model: trained model
    :param X: test instances
    :param y: test labels
    :return:
    '''
    print("\nTesting the model...")
    for i in X:
        predict = model.predict(i, batch_size= batch_size)
        print('Prediction of the model: '.format(np.argmax(predict)))


# @jit
# def generate(X_tr):
#
#     return generated


