from tensorflow.contrib.keras import applications, layers, models, optimizers
import numpy as np
from numba import jit
# from PIL import Image as im

@jit
def loadModel(n_classes):
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
def trainModel(model, X, y):
    batch_size = 32
    n_epoch = 1
    model.fit(x= X, y= y, batch_size= batch_size, epochs= n_epoch, verbose=0)

    return model

@jit
def testModel(model, X):
    batch_size = 32
    for i in X:
        predict = model.predict(i, batch_size= batch_size)
        print('Prediction of the model: '.format(np.argmax(predict)))


