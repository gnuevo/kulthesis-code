"""Simple autoencoder with generator

This is a simple autoencdoer (it has only one compressed layer). It uses a 
generator to iterate over the data in an online fashion. This allows two 
interesting things:

    1. Changing the size of the samples easily; and
    2. a method for data augmentation (look for the step argument)

"""
from keras.layers import Input, Dense, Reshape
from keras.models import Model
from generator.SimpleAutoencoderGenerator import SimpleAutoencoderGenerator
from functools import reduce
import operator

class AutoencoderWithGenerator(object):
    def __init__(self, hdf5_file, hdf5_group, input_dimension=(1000, 2),
                 encoding_dim=32):
        self.__generator = SimpleAutoencoderGenerator(hdf5_file, hdf5_group)
        self.__input_dimension = input_dimension # equivalent to sample_length
        self.__encoding_dim = encoding_dim

        # total_audio_samples =

        self.initialise(input_dimension=input_dimension, encoding_dim=encoding_dim)

    def initialise(self, input_dimension=(1000, 2), encoding_dim=32):
        if len(input_dimension) > 1:
            flattened_dimension = (reduce(operator.mul, input_dimension),)
        else:
            flattened_dimension = input_dimension
        print("flatten dim", flattened_dimension)
        # this is the size of our encoded representations
        # this is our input placeholder
        input_tensor = Input(shape=input_dimension)
        print("here",input_tensor._keras_shape)

        # reshape if necessary, encode
        if len(input_dimension) > 1:
            input_reshape = Reshape(flattened_dimension,
                                    input_shape=input_dimension)(
                input_tensor)
            encoded = Dense(encoding_dim, activation='relu')(input_reshape)
        else:
            encoded = Dense(encoding_dim, activation='relu')(input_tensor)

        # decode, reshape if necessary
        if len(input_dimension) > 1:
            decoded = Dense(flattened_dimension[0], activation='relu')(encoded)
            output = Reshape(input_dimension,
                             input_shape=flattened_dimension)(decoded)
        else:
            decoded = Dense(flattened_dimension, activation='relu')(encoded)
            output = decoded


        # # "encoded" is the encoded representation of the input
        # encoded = Dense(encoding_dim, activation='relu')(input_tensor)
        # # "decoded" is the lossy reconstruction of the input
        # decoded = Dense(input_dimension[0]*input_dimension[1],
        #                 activation='sigmoid')(encoded)

        # this model maps an input to its reconstruction
        self.__autoencoder = Model(input_tensor, output)
        self.__autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    def fit_generator(self, batch_size=1000, epochs=1, step=None):
        if step == None: step = self.__input_dimension[0]
        print(self.__generator.get_nsamples(self.__input_dimension[0], step,
                                            batch_size))
        self.__autoencoder.fit_generator(
            self.__generator.generate_samples(self.__input_dimension[0],
                                              step=step, batch_size=batch_size),
            self.__generator.get_nsamples(self.__input_dimension[0], step,
                                          batch_size),
            epochs=epochs
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Autoencoder with generator")
    # arguments for gather_dataset_online
    parser.add_argument("hdf5_file", help="File containing the dataset file", type=str)
    parser.add_argument("--hdf5-group", help="Group name with the dataset", type=str)
    parser.add_argument("--dim", help="Input dimensions. Number of audio "
                                      "samples per sample", type=int,
                        required=True)
    parser.add_argument("--encoded_size", help="The encoded size", type=int,
                        default=32)
    dargs = parser.parse_args()

    hdf5_file = dargs.hdf5_file
    hdf5_group = dargs.hdf5_group
    dimension = dargs.dim
    encoded_size = dargs.encoded_size

    autoencoder = AutoencoderWithGenerator(hdf5_file, hdf5_group,
                                           input_dimension=(dimension, 2),
                                           encoding_dim=encoded_size)

