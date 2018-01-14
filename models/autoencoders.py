"""This file is meant to store autoencoder models

"""
from keras.layers import Input, Dense, Reshape
from keras.models import Model
from functools import reduce
import operator
try:
   import cPickle as pickle
except:
   import pickle
from dataset.dataset import Dataset
from generator.readers import Reader
from generator.batchers import DoubleSynchronisedRandomisedBatcher as \
    DSRBatcher
from .simple_autoencoder_with_generator import SimpleAutoencoderGenerator
from .simple_autoencoder_with_generator import AutoencoderWithGenerator

def SimpleAutoencoder(input_dimension=(1000, 2), encoding_dim=32,
                         activation='relu', optimizer='adadelta',
                         loss='binary_crossentropy'):
    """This function creates and returns a simple autoencoder

    Args:
        input_dimension (int or tuple of ints): input dimension for the 
                    network, if it is a tuple the input is flattened to only 
                    one dimension
        encoding_dim (int): compressed size
        activation: activation function
        optimizer: optimizer
        loss: loss

    Returns:

    """
    #FIXME I think this class has never been tried BE CAREFULL
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
        encoded = Dense(encoding_dim, activation=activation,
                        name="encoder")(
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


class DoubleAutoencoderGenerator(AutoencoderWithGenerator):
    """
    
    """

    def __init__(self, input_file, input_group, output_file,
                 output_group, input_dimension=(1000, 2), encoding_dim=32):
        """Initialises the autoencoder

        This is a simple autoencoder. It has only one layer for encoding and 
        one layer for decoding. The input and output can be different

        Args:
            hdf5_file (str): name of the hdf5 file that contains the dataset 
            hdf5_group (str): name of the hdf5 group that contains the dataset
            input_dimension: (int or tuple of ints) input dimension of the 
                network, it is equivalent to sample length
            encoding_dim (int): size of the encoded layer of the autoencoder 
        """
        self.__input_file = input_file
        self.__input_group = input_group
        self.__output_file = output_file
        self.__output_group = output_group
        self.__input_dimension = input_dimension
        self.__encoding_dim = encoding_dim
        if len(input_dimension) >= 2:
            self.__sample_length = input_dimension[0]
        elif len(input_dimension) == 1:
            self.__sample_length = input_dimension

        self.__initialise(input_dimension=input_dimension,
                          encoding_dim=encoding_dim)


    def create_generator(self, input_file, input_group,
                         output_file, output_group, sample_length,
                         step=None, section=None):
        """
        
        Args:
            input_file: 
            input_group: 
            output_file: 
            output_group: 
            sample_length: length of the desired samples (first value of 
            input_dimension)
            step: 
            section: 

        Returns:
        """
        # get datasets
        input_dataset = Dataset(input_file)
        output_dataset = Dataset(output_file)

        # create readers
        input_reader = Reader(input_dataset.group(input_group).data(),
                              sample_length, input_dataset.group(
                input_group).chunk_size, step, section)
        output_reader = Reader(output_dataset.group(output_group).data(),
                               sample_length, output_dataset.group(
                output_group).chunk_size, step, section)

        # create batcher
        batcher = DSRBatcher(input_reader, output_reader)
        return batcher


    def initialise(self, activation='relu', optimizer='adadelta',
                   loss='binary_crossentropy'):
        """Initialise the neural network

        The neural network gets already initialised during __init__ so if 
        this method is called the network behind is restarted.
        """
        # call to the parent method
        super().__initialise(input_dimension=self.__input_dimension,
                          encoding_dim=self.__encoding_dim,
                          activation=activation, optimizer=optimizer,
                          loss=loss)


    def fit_generator(self, batch_size=100, epochs=1, step=None,
                      train_section=None, val_section=None,
                      history_file=None):
        if step == None: step = self.__input_dimension[0]
        if val_section:
            val_generator = self.create_generator(self.__input_file,
                                                  self.__input_group,
                                                  self.__output_file,
                                                  self.__output_group,
                                                  sample_length=self.__sample_length,
                                                  step=step,
                                                  section=val_section)

            val_data = val_generator.generate_batches(batch_size=batch_size,
                                                      randomise=False)
            val_steps = val_generator.get_nbatches_in_epoch()
        else:
            val_data = None
            val_steps = None

        callbacks = self.__callbacks if not self.__callbacks == [] else None
        print("Fit generator, train section", train_section)
        print("validation_steps", val_steps)
        train_generator = self.create_generator(self.__input_file,
                                           self.__input_group,
                                           self.__output_file,
                                           self.__output_group,
                                           sample_length=self.__sample_length,
                                           step=step,
                                           section=train_section)
        train_data = train_generator.generate_batches(batch_size=batch_size,
                                                      randomise=True)
        train_steps = train_generator.get_nbatches_in_epoch()
        print(self.__generator.get_nbatches_in_epoch())
        history = super().__autoencoder.fit_generator(
            train_data.generate_samples(),
            train_steps,
            validation_data=val_data,
            validation_steps=val_steps,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
            )
        if history_file is not None:
            pickle(history.history, history_file)