"""
Network building example code for REU 2022 MNIST example

Jay Rothenberger (jay.c.rothenberger@ou.edu)

"""


import tensorflow as tf
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Dense, Input, Concatenate
from time import time


def build_sequential_model(conv_filters,
                           conv_size,
                           dense_layers,
                           image_size=(28, 28, 1),
                           learning_rate=1e-3,
                           n_classes=10,
                           activation='selu'):
    # create the model object
    model = tf.keras.Sequential()
    # add an input layer (this step is only needed for the summary)
    model.add(Input(image_size))
    # add the convolutional layers
    for (filters, kernel) in zip(conv_filters, conv_size):
        model.add(Conv2D(filters=filters, kernel_size=(kernel, kernel), activation=activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # flatten
    model.add(Flatten())
    # add dense layers
    for neurons in dense_layers:
        model.add(Dense(neurons, activation=activation))
    # classification output
    model.add(Dense(n_classes, activation=tf.keras.activations.softmax))
    # optimizer
    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)
    # compile the model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['sparse_categorical_accuracy'])

    # Generate an ASCII representation of the architecture
    print(model.summary())

    return model


def build_functional_model(conv_filters,
                           conv_size,
                           dense_layers,
                           image_size=(28, 28, 1),
                           learning_rate=1e-3,
                           n_classes=10,
                           activation='selu'):
    # define the input layer (required)
    inputs = Input(image_size)
    # set reference x separately to keep track of the input layer
    x = inputs
    # construct the convolutional part
    for (filters, kernel) in zip(conv_filters, conv_size):
        # each layer is a function of the previous layer, we can reuse reference x
        x = Conv2D(filters=filters, kernel_size=(kernel, kernel), activation=activation)(x)
        # pooling after a convolution (or two) is a standard simple technique
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    # flatten
    x = Flatten()(x)
    # construct the dense part
    for neurons in dense_layers:
        x = Dense(neurons, activation=activation)(x)
    # classification output
    outputs = Dense(n_classes, activation=tf.keras.activations.softmax)(x)
    # optimizer
    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)
    # when we compile the model we must specify inputs and outputs
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs],
                           name=f'cnn_model_{"%02d" % time()}')
    # compile the model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['sparse_categorical_accuracy'])

    # Generate an ASCII representation of the architecture
    print(model.summary())

    return model


def build_parallel_functional_model(conv_filters,
                                    conv_size,
                                    dense_layers,
                                    image_size=(28, 28, 1),
                                    learning_rate=1e-3,
                                    n_classes=10,
                                    activation='selu'):
    # define the input tensor
    inputs = Input(image_size)

    x = inputs
    # construct the convolutional block
    for (filters, kernel) in zip(conv_filters, conv_size):
        # here we keep track of the input of each block
        ins = x
        # there are two paths through which the data and gradient can flow
        # 1st path is x:
        x = Conv2D(filters=filters, kernel_size=(kernel, kernel), activation=activation, padding='same')(ins)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        # 2nd path is y:
        y = Conv2D(filters=filters, kernel_size=(1, 1), activation=activation, padding='same')(ins)
        y = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(y)
        # both paths' outputs are concatenated across the filter dimension
        x = Concatenate()([x, y])
        # and then an additional convolution that reduces the total filter dimension
        # is performed
        x = Conv2D(filters=filters, kernel_size=(1, 1), activation=activation)(x)

    # flatten
    x = Flatten()(x)
    # construct the dense part
    for neurons in dense_layers:
        x = Dense(neurons, activation=activation)(x)
    # classification output
    outputs = Dense(n_classes, activation=tf.keras.activations.softmax)(x)

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)
    # build the model
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs],
                           name=f'cnn_model_{"%02d" % time()}')
    # compile the model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['sparse_categorical_accuracy'])

    # Generate an ASCII representation of the architecture
    print(model.summary())

    return model
