"""
This file contains the simplest autoencoder, as defined in https://blog.keras.io/building-autoencoders-in-keras.html
"""
from keras.layers import Input, Dense
from keras.models import Model
import h5py
import numpy as np

BIG_CHUNKS = 800000 # how many samples we take from the dataset

class Autoencoder(object):

    def __init__(self):
        self.autoencoder = None

    def initialise(self, input_dimension=(1000, 2), encoding_dim=32):
        # this is the size of our encoded representations
        # this is our input placeholder
        input_img = Input(shape=input_dimension)
        # "encoded" is the encoded representation of the input
        encoded = Dense(encoding_dim, activation='relu')(input_img)
        # "decoded" is the lossy reconstruction of the input
        decoded = Dense(2000, activation='sigmoid')(encoded)

        # this model maps an input to its reconstruction
        self.autoencoder = Model(input_img, decoded)
        self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    def fit(self, train_data, test_data):
        try:
            hist = self.autoencoder.fit(train_data, train_data,
                    epochs=4,
                    batch_size=4096,  # after 2048 the average time per epoch (0.8M elements) is 20s
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
    print(data.shape)
    flattened_data = np.reshape(data, (BIG_CHUNKS, sample_size * audio_channels))

    autoencoder = Autoencoder()
    # autoencoder.initialise()
    autoencoder.initialise(input_dimension=(sample_size*audio_channels,))
    autoencoder.fit(flattened_data, _)