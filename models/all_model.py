import keras
from keras import backend as K
from keras.layers.core import Lambda
from keras.regularizers import l2

from keras.models import Sequential
def googlenet(n_classes,IMG_SIZE,dataset,use_imagenet=True):
    # load pre-trained model graph, don't add final layer
    model = keras.applications.InceptionV3(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                          weights='imagenet' if use_imagenet else None)
    # add global pooling just like in InceptionV3
    new_output = keras.layers.GlobalAveragePooling2D()(model.output)
    # add new dense layer for our labels
    new_output = keras.layers.Dense(n_classes, activation='softmax',name="fc_"+dataset)(new_output)
    model = keras.engine.training.Model(model.inputs, new_output)
    return model

def vgg(n_classes,IMG_SIZE,dataset,use_imagenet=True):
    # load pre-trained model graph, don't add final layer
    model = keras.applications.VGG16(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                          weights='imagenet' if use_imagenet else None)
    # add global pooling just like in InceptionV3
    new_output = keras.layers.Flatten(name='flatten')(model.output)
    new_output =   keras.layers.Dense(4096, activation='relu', name='fc1')(new_output)
    
    new_output = keras.layers.Dense(4096, activation='relu', name='fc2')(new_output)
    
    # add new dense layer for our labels
    new_output = keras.layers.Dense(n_classes, activation='softmax',name='fc_'+dataset)(new_output)
    model = keras.engine.training.Model(model.inputs, new_output)
    return model    
def resnet(n_classes,IMG_SIZE,dataset,use_imagenet=True):
    model = keras.applications.ResNet50(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                          weights='imagenet' if use_imagenet else None)
    # add global pooling just like in InceptionV3
    new_output = keras.layers.GlobalAveragePooling2D(name='avg_pool')(model.output)
    # add new dense layer for our labels
    new_output = keras.layers.Dense(n_classes, activation='softmax',name="fc"+dataset)(new_output)
    model = keras.engine.training.Model(model.inputs, new_output)
    return model


def Alexnet_initialization(shape, name=None):

    import numpy as np
    from keras import backend as K

    mu, sigma = 0, 0.01
    return K.variable(np.random.normal(mu, sigma, shape), name=name)

def Alexnet(n_classes,IMG_SIZE,dataset,use_imagenet=True):

    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation, Flatten
    from keras.layers.convolutional import Convolution2D, MaxPooling2D
    from keras.layers.normalization import BatchNormalization
    from keras.regularizers import l2

    weight_decay = 0.0005

    model = Sequential()

    # Conv1
    model.add(Convolution2D(nb_filter=96, nb_row=11, nb_col=11, border_mode='valid', input_shape=(IMG_SIZE, IMG_SIZE,3),
                            init=Alexnet_initialization, subsample=(4, 4),
                            W_regularizer=l2(weight_decay)))  # subsample is stride
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    # Conv2
    model.add(
        Convolution2D(256, 5, 5, border_mode='same', init=Alexnet_initialization, W_regularizer=l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    # Conv3
    model.add(
        Convolution2D(384, 3, 3, border_mode='same', init=Alexnet_initialization, W_regularizer=l2(weight_decay)))
    model.add(Activation('relu'))

    # Conv4
    model.add(
        Convolution2D(384, 3, 3, border_mode='same', init=Alexnet_initialization, W_regularizer=l2(weight_decay)))
    model.add(Activation('relu'))

    # Conv5
    model.add(
        Convolution2D(256, 3, 3, border_mode='same', init=Alexnet_initialization, W_regularizer=l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())

    # Fc6
    model.add(Dense(4096, init=Alexnet_initialization, W_regularizer=l2(weight_decay)))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))

    # Fc7
    model.add(Dense(4096, init=Alexnet_initialization, W_regularizer=l2(weight_decay)))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))

    # Fc8
    model.add(Dense(n_classes, init=Alexnet_initialization, W_regularizer=l2(weight_decay),name="fc"+dataset))
    model.add(Activation('softmax'))

    return model


