"""Networks

This file stores functions that return networks. In this sense it's easy, for
example, to write a function that creates an autoencoder and returns it. So 
the generation of the autoencoder is totally split from the functions that 
make use of it.
"""
from keras.layers import Input, Dense, Reshape
from keras.models import Model
from functools import reduce
import operator


def SimpleAutoencoderNetwork(input_dimension=(1000, 2), encoding_dim=32,
                 activation='relu', optimizer='adadelta',
                 loss='binary_crossentropy'):
    """Initialises the neural network

    Creates an autoencoder type neural network following the 
    specifications of the arguments

    Args:
        input_dimension (int or tuple of ints): input dimension for the 
            network, if it is a tuple the input is flattened to only 
            one dimension
        encoding_dim (int): compressed size
    """
    if len(input_dimension) > 1:
        flattened_dimension = (reduce(operator.mul, input_dimension),)
    else:
        flattened_dimension = input_dimension
    # this is the size of our encoded representations
    # this is our input placeholder
    input_tensor = Input(shape=input_dimension)

    # reshape if necessary, encode
    if len(input_dimension) > 1:
        input_reshape = Reshape(flattened_dimension,
                                input_shape=input_dimension,
                                name="input_reshape")(
            input_tensor)
        encoded = Dense(encoding_dim, activation=activation,
                        name="encoder")(input_reshape)
    else:
        encoded = Dense(encoding_dim, activation=activation, name="encoder")(
            input_tensor)

    # decode, reshape if necessary
    if len(input_dimension) > 1:
        decoded = Dense(flattened_dimension[0], activation="tanh",
                        name="decoder"
                        )(encoded)
        output = Reshape(input_dimension,
                         input_shape=flattened_dimension,
                         name="output_reshape")(
            decoded)
    else:
        decoded = Dense(flattened_dimension[0], activation="tanh",
                        name="decoder")(encoded)
        output = decoded

    # this model maps an input to its reconstruction
    autoencoder = Model(input_tensor, output)
    autoencoder.compile(optimizer=optimizer, loss=loss)

    return autoencoder


def DeepAutoencoderNetwork(input_dimension=(1000, 2),
                           middle_layers=[], encoding_dim=32,
                 activation='relu', optimizer='adadelta',
                 loss='binary_crossentropy'):
    """Initialises the neural network

    Creates an autoencoder type neural network following the 
    specifications of the arguments

    Args:
        input_dimension (int or tuple of ints): input dimension for the 
            network, if it is a tuple the input is flattened to only 
            one dimension
        encoding_dim (int): compressed size
    """
    if len(input_dimension) > 1:
        flattened_dimension = (reduce(operator.mul, input_dimension),)
    else:
        flattened_dimension = input_dimension
    print(flattened_dimension)
    # this is the size of our encoded representations
    # this is our input placeholder
    input_tensor = Input(shape=input_dimension)
    # reshape if necessary
    if len(input_dimension) > 1:
        input = Reshape(flattened_dimension, input_shape=input_dimension,
                        name="input_reshaped")
    else:
        input = input_tensor


    # encoder section
    previous_layer = input
    for i, layer_size in enumerate(middle_layers):
        name = "encoder{}".format(i)
        new_layer = Dense(layer_size, activation=activation, name=name)(previous_layer)
        previous_layer = new_layer

    # encoded layer
    encoded = Dense(encoding_dim, activation=activation, name="encoder")(
        previous_layer)

    # decoder section
    # reverse middle layers
    middle_layers.reverse()
    indices = list(range(len(middle_layers)))
    indices.reverse()
    previous_layer = encoded
    for i, layer_size in zip(indices, middle_layers):
        name = "decoder{}".format(i)
        new_layer = Dense(layer_size, activation=activation, name=name)(previous_layer)
        previous_layer = new_layer

    # output
    if len(input_dimension) > 1:
        decoded = Dense(flattened_dimension[0], activation="tanh",
                       name="output")(previous_layer)
        output = Reshape(input_dimension,
                         input_shape=flattened_dimension,
                         name="output_reshape")(decoded)
    else:
        decoded = Dense(flattened_dimension[0], activation="tanh",
                        name="output")(previous_layer)
        output = decoded

    # this model maps an input to its reconstruction
    autoencoder = Model(input_tensor, output)
    autoencoder.compile(optimizer=optimizer, loss=loss)

    print(autoencoder.summary())

    return autoencoder