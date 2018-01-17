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
    print("flatten dim", flattened_dimension)
    # this is the size of our encoded representations
    # this is our input placeholder
    input_tensor = Input(shape=input_dimension)
    print("here", input_tensor._keras_shape)

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
        decoded = Dense(flattened_dimension[0], activation=activation,
                        name="decoder"
                        )(encoded)
        output = Reshape(input_dimension,
                         input_shape=flattened_dimension,
                         name="output_reshape")(
            decoded)
    else:
        decoded = Dense(flattened_dimension, activation=activation,
                        name="decoder")(encoded)
        output = decoded

    # this model maps an input to its reconstruction
    autoencoder = Model(input_tensor, output)
    autoencoder.compile(optimizer=optimizer, loss=loss)

    return autoencoder