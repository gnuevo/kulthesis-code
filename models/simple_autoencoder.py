"""
This file contains the simplest autoencoder, as defined in https://blog.keras.io/building-autoencoders-in-keras.html
"""
from keras.layers import Input, Dense, Reshape
from keras.models import Model
from keras.engine.topology import InputLayer
import h5py
import numpy as np

BIG_CHUNKS = 20000 # how many samples we take from the dataset

class Autoencoder(object):

    def __init__(self):
        self.autoencoder = None

    def initialise(self, input_dimension=(5000, 2), encoding_dim=32):
        # this is the size of our encoded representations
        # this is our input placeholder
        print()
        print("input dimension", input_dimension)
        input_img = Input(shape=input_dimension)
        print(type(input_img))
        print("input shape",input_img._keras_shape)

        # reshape the input from a vector to an input
        input_reshape = Reshape((input_dimension[0]*input_dimension[1],),
                                input_shape=input_img._keras_shape[1:])(input_img)
        print("input reshape shape",input_reshape.shape)

        # "encoded" is the encoded representation of the input
        encoded = Dense(encoding_dim, activation='relu')(input_reshape)
        print(type(encoded))
        print("encoded shape", encoded.shape)
        # "decoded" is the lossy reconstruction of the input
        decoded = Dense(input_dimension[0]*input_dimension[1],
                        activation='sigmoid')(
            encoded)

        # reshape output
        output_reshape = Reshape(input_dimension,
                                 input_shape=(input_dimension[
                                                  0]*input_dimension[1],))(decoded)
        print("decoded shape", decoded.shape)

        # this model maps an input to its reconstruction
        self.autoencoder = Model(input_img, output_reshape)
        # print("model config", self.autoencoder.get_config())
        for layer in self.autoencoder.get_config()['layers']:
            print(layer)
            print()
        print(self.autoencoder.layers)
        for layer in self.autoencoder.layers:
            print(layer.output_shape)
        # for layer in self.autoencoder.layers:
        #     print(type(layer))
        #     print(layer.__dict__)
        #     if isinstance(layer, InputLayer):
        #         # print(layer.name, layer._keras_shape)
        #         pass
        #     else:
        #         print(layer.name, layer.shape)
        self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    def fit(self, train_data, test_data):
        try:
            print("train data shape", train_data.shape)
            hist = self.autoencoder.fit(train_data, train_data,
                    epochs=100,
                    batch_size=2048,  # after 2048 the average time per epoch (
                                        # 0.8M elements) is 20s
                    shuffle=True)
            print(hist.history)
        except UnboundLocalError as e:
            print("Nasty stupid UnboundLocalError")

def read_dataset(dataset_file, group):
    """
    Reads and returns the dataset
    :param dataset_file: 
    :param group: 
    :return: train data, test data (currently test data = None)
    """
    h5f = h5py.File(dataset_file, 'r')
    try:
        h5g = h5f[group]
    except:
        print("Error while getting the hdf5 group. Probably it doesn't exist")
        exit(-1)
    h5d = h5g['data'] # get dataset
    dataset = h5d[:BIG_CHUNKS]
    h5f.close()
    return dataset, None

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Autoencoder")
    # arguments for gather_dataset_online
    parser.add_argument("hdf5_file", help="File containing the dataset file", type=str)
    parser.add_argument("--hdf5-group", help="Group name with the dataset", type=str)
    dargs = parser.parse_args()

    hdf5_file = dargs.hdf5_file
    hdf5_group = dargs.hdf5_group

    data, _ = read_dataset(hdf5_file, hdf5_group)
    _, sample_size, audio_channels = data.shape
    print("data shape", data.shape)
    flattened_data = np.reshape(data, (BIG_CHUNKS, sample_size,audio_channels))
    print("data shape after flatten", data.shape)

    autoencoder = Autoencoder()
    # autoencoder.initialise()
    autoencoder.initialise(input_dimension=(sample_size,audio_channels))
    autoencoder.fit(flattened_data, _)