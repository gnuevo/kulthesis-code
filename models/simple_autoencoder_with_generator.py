"""Simple autoencoder with generator

This is a simple autoencdoer (it has only one compressed layer). It uses a 
generator to iterate over the data in an online fashion. This allows two 
interesting things:

    1. Changing the size of the samples easily; and
    2. a method for data augmentation (look for the step argument)

"""
from keras.layers import Input, Dense, Reshape
from keras.models import Model
from keras.callbacks import TensorBoard
from generator.SimpleAutoencoderGenerator import SimpleAutoencoderGenerator
from functools import reduce
import operator
import h5py

class AutoencoderWithGenerator(object):
    def __init__(self, hdf5_file, hdf5_group, input_dimension=(1000, 2),
                 encoding_dim=32):
        """Initialises the autoencoder
        
        This is a simple autoencoder. It has only one layer for encoding and 
        one layer for decoding.
        
        Args:
            hdf5_file (str): name of the hdf5 file that contains the dataset 
            hdf5_group (str): name of the hdf5 group that contains the dataset
            input_dimension: (int or tuple of ints) input dimension of the 
                network
            encoding_dim (int): size of the encoded layer of the autoencoder 
        """
        self.__hdf5_file = hdf5_file
        self.__hdf5_group = hdf5_group
        self.__generator = SimpleAutoencoderGenerator(hdf5_file, hdf5_group)
        self.__input_dimension = input_dimension # equivalent to sample_length
        self.__encoding_dim = encoding_dim
        # list to store the callbacks, initialised empty
        self.__callbacks = []

        self.__initialise(input_dimension=input_dimension, encoding_dim=encoding_dim)

    def initialise(self):
        """Initialise the neural network
        
        The neural network gets already initialised during __init__ so if 
        this methode is called the network behind is restarted.
        """
        self.__initialise(input_dimension=self.__input_dimension,
                          encoding_dim=self.__encoding_dim)

    def __initialise(self, input_dimension=(1000, 2), encoding_dim=32):
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
        print("here",input_tensor._keras_shape)

        # reshape if necessary, encode
        if len(input_dimension) > 1:
            input_reshape = Reshape(flattened_dimension,
                                    input_shape=input_dimension,
                                    name="input_reshape")(
                input_tensor)
            encoded = Dense(encoding_dim, activation='relu',
                            name="encoder")(input_reshape)
        else:
            encoded = Dense(encoding_dim, activation='relu', name="encoder")(
                input_tensor)

        # decode, reshape if necessary
        if len(input_dimension) > 1:
            decoded = Dense(flattened_dimension[0], activation='relu',
                            name="decoder"
                            )(encoded)
            output = Reshape(input_dimension,
                             input_shape=flattened_dimension,
                             name="output_reshape")(
                decoded)
        else:
            decoded = Dense(flattened_dimension, activation='relu',
                            name="decoder")(encoded)
            output = decoded

        # this model maps an input to its reconstruction
        self.__autoencoder = Model(input_tensor, output)
        self.__autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    def callback_add_tensorboard(self, log_dir='/tmp/autoencoder',
                                 histogram_freq=1, write_graph=True):
        from callbacks.customTensorBoard import customTensorBoard
        tensorboard_callback = TensorBoard(log_dir=log_dir,
                                           histogram_freq=histogram_freq,
                                           write_graph=write_graph)
        # remove a posible tensorboard callback from the list
        # there is no point into keeping several callbacks to tensorflow
        self.__callbacks = [callback for callback in self.__callbacks if not
                        type(callback) == type(tensorboard_callback)]
        self.__callbacks.append(tensorboard_callback)

        tensorboard_callback = customTensorBoard(log_dir=log_dir,
                                           histogram_freq=histogram_freq,
                                           write_graph=write_graph)
        # remove a posible tensorboard callback from the list
        # there is no point into keeping several callbacks to tensorflow
        self.__callbacks = [callback for callback in self.__callbacks if not
        type(callback) == type(tensorboard_callback)]
        self.__callbacks.append(tensorboard_callback)

    def fit_generator(self, batch_size=1000, epochs=1, step=None,
                      train_section=None, val_section=None):
        if step == None: step = self.__input_dimension[0]
        if val_section:
            val_generator = SimpleAutoencoderGenerator(self.__hdf5_file,
                                                       self.__hdf5_group)
            val_generator.configure(sample_length=self.__input_dimension[0],
                                   batch_size=batch_size,
                                   step=step,
                                   section=val_section)
            val_data = val_generator.generate_samples()
            val_steps = val_generator.get_nbatches_in_epoch()
        else:
            val_data = None
            val_steps = None

        callbacks = self.__callbacks if not self.__callbacks == [] else None

        self.__generator.configure(sample_length=self.__input_dimension[0],
                                   batch_size=batch_size,
                                   step=step,
                                   section=train_section)
        print(self.__generator.get_nbatches_in_epoch())
        self.__autoencoder.fit_generator(
            self.__generator.generate_samples(),
            self.__generator.get_nbatches_in_epoch(),
            validation_data=val_data,
            validation_steps=val_steps,
            epochs=epochs,
            callbacks=callbacks
            )

    def train_dataset(self, batch_size=1000, epochs=1, step=None,
                      validation=True):
        """Trains the dataset using the train section
        
        Gets the train section from the dataset and trains it with that section
        
        Args:
            batch_size (int): size of the batch
            epochs (int): number of epochs
            step (int): offset between samples
        """
        if step == None: step = self.__input_dimension[0]

        # get info of the training section
        data = self.__generator.h5g["data"]
        train_section = data.attrs["train_set"]
        train_section = tuple(train_section)

        if validation:
            val_section = data.attrs["val_set"]
            val_section = tuple(val_section)
        else:
            val_section = None

        # train
        self.fit_generator(batch_size=batch_size, epochs=epochs, step=step,
                           train_section=train_section, val_section=val_section)

    def validate_dataset(self):
        pass
    def test_dataset(self):
        pass


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

