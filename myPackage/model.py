from tensorflow.contrib.keras import applications, layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import model_from_json
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, roc_auc_score
import numpy as np
from numba import jit
from matplotlib import pyplot as plt
from myPackage import tools as tl
from os.path import basename, splitext

@jit
def loadModel(n_classes, trainable= None):
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

    if trainable is not None:
        for layer in base_model.layers[:trainable]:
            layer.trainable = False
        for layer in base_model.layers[trainable:]:
            layer.trainable = True
        opt = optimizers.RMSprop(lr=0.0001)
    else:
        for layer in base_model.layers:
            layer.trainable = False
        opt = optimizers.RMSprop(lr=0.001)

    model.compile(loss= 'sparse_categorical_crossentropy', optimizer= opt, metrics= ['accuracy'])

    return model


# @jit
# def trainGenModel(model, batch_size, n_epoch, X, y, val_split= None):
#     '''
#     Train the loaded model
#     :param model: model loaded
#     :param X: training instances
#     :param y: training labels
#     :return: trained model
#     '''
#     print("\nTraining model with data augmentation...")
#     datagen = ImageDataGenerator(
#         rotation_range=40,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         vertical_flip=True,
#         fill_mode='nearest')
#     # generated = datagen.fit(X)
#     if val_split != None:
#         history = model.fit_generator(datagen.flow(X, y, batch_size= batch_size),
#                     steps_per_epoch=len(X) / batch_size, epochs= n_epoch, use_multiprocessing= True)
#
#         #  "Accuracy"
#         plt.plot(history.history['acc'])
#         plt.plot(history.history['val_acc'])
#         plt.title('Model accuracy')
#         plt.ylabel('Accuracy')
#         plt.xlabel('Epoch')
#         plt.legend(['Train', 'Validation'], loc='upper left')
#         plt.show()
#         # "Loss"
#         plt.plot(history.history['loss'])
#         plt.plot(history.history['val_loss'])
#         plt.title('Model loss')
#         plt.ylabel('Loss')
#         plt.xlabel('Epoch')
#         plt.legend(['Train', 'Validation'], loc='upper left')
#         plt.show()
#     else:
#         history = model.fit_generator(datagen.flow(X, y, batch_size=batch_size),
#                                       steps_per_epoch=len(X) / batch_size, epochs=n_epoch, workers= -1)
#         #  "Accuracy"
#         plt.plot(history.history['acc'])
#         plt.title('Model accuracy')
#         plt.ylabel('Accuracy')
#         plt.xlabel('Epoch')
#         plt.legend(['Train'], loc='upper left')
#         plt.show()
#         # "Loss"
#         plt.plot(history.history['loss'])
#         plt.title('Model loss')
#         plt.ylabel('Loss')
#         plt.xlabel('Epoch')
#         plt.legend(['Train'], loc='upper left')
#         plt.show()
#     # if val_split != 0.0:
#     #     plt.plot(history.history['val_acc'])
#     #     plt.show()
#     #for i, val in enumerate(len(history)):
#        # print("Epoch {} has {} accuracy".format(i, history[i]))
#     print(history)
#     return model

@jit
def trainModel(model, batch_size, n_epoch, X, y, val_split= None, generator= None):
    '''
    Train the loaded model
    :param model: model loaded
    :param X: training instances
    :param y: training labels
    :return: trained model
    '''

    if val_split is not None:
        if generator:
            print("\nTraining model with validation split and data augmentation...")
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
            history = model.fit_generator(datagen.flow(X, y, batch_size=batch_size),
                                          steps_per_epoch=len(X) / batch_size, epochs=n_epoch, workers= 4)
        else:
            print("\nTraining model with validation split...")
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
        if generator:
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
            history = model.fit_generator(datagen.flow(X, y, batch_size=batch_size),
                                          steps_per_epoch=len(X) / batch_size, epochs=n_epoch, workers= 4)
        else:
            print("\nTraining model...")
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
    #for i, val in enumerate(len(history)):
    #    print("Epoch {} has {} accuracy".format(i, history[i]))

    # pickle.dump(model, open(modelName, 'wb'))
    print(history)
    print("Training finished!!")
    return model


@jit
def testModel(json, weights, batch_size, X_te, Y_te):
    '''
    Test the model
    :param model: trained model
    :param X: test instances
    :param y: test labels
    :return:
    '''
    # load json and create model
    print("Loading model...")
    json_file = open(json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights)
    model_str = basename(json)
    print("Loaded model from disk '{}'\n".format(model_str))

    # evaluate loaded model on test data
    if '249' in model_str:
        # opt = optimizers.RMSprop(lr=0.0001)
        print("Evaluating model with data augmentation and fine tuning...")
    elif '0.2' in model_str:
        print("Evaluating model with validation data...")
    elif 'True_None' in model_str:
        print("Evaluating model with data augmentation...")
    elif 'False_None' in model_str:
        print("Evaluating baseline model...")

    loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    predict = []
    for i in X_te:
        predict.append(np.argmax(loaded_model.predict(np.expand_dims(i, axis=0), batch_size=batch_size)))
        # loaded_model.predict(np.expand_dims(i, axis=0), batch_size=batch_size)
        # print('Prediction of the model: '.format(np.argmax(predict)))
    predict = np.array(predict)

    accuracy = accuracy_score(Y_te, predict) * 100
    precision_macro = precision_score(Y_te, predict, average='macro')* 100
    precision_micro = precision_score(Y_te, predict, average='micro') * 100
    recall_macro = recall_score(Y_te, predict, average='macro') * 100
    recall_micro = recall_score(Y_te, predict, average='micro') * 100
    print("Accuracy: {:.3f}%\n"
          "Precision (macro): {:.3f}%\n"
          "Precision (micro): {:.3f}%\n"
          "Recall (macro): {:.3f}%\n"
          "Recall (micro): {:.3f}%\n".format(accuracy, precision_macro, precision_micro, recall_macro, recall_micro))

    # Compute and plot normalized confusion matrix
    # cnf_matrix = confusion_matrix(Y_te, predict)
    # tl.plot_confusion_matrix(cnf_matrix, classes= Y_te, normalize=True,
    #                          title="Normalized confusion matrix")
    print("Evaluation finished!!\n\n")

def saveModel(nameModel, trainedModel):
    # serialize model to JSON
    model_json = trainedModel.to_json()
    with open(nameModel+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    trainedModel.save_weights(nameModel+".h5")
    print("Saved model to disk")

@jit
def getMetrics(cm):
    num_classes = len(cm)
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    TN = []
    for i in range(num_classes):
        temp = np.delete(cm, i, 0)  # delete ith row
        temp = np.delete(temp, i, 1)  # delete ith column
        TN.append(sum(sum(temp)))
    return TP, FP, FN, TN