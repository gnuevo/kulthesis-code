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
from keras.callbacks import TensorBoard
from .networks import SimpleAutoencoderNetwork


class AutoencoderSkeleton(object):
    """Contains basic functionality common to all autoencoders
    
    Holds those methods that are used by all the autoencoders, for example 
    those related to train, test and callbacks. This should help to 
    modularity and reusability of the code.
    """

    def __init__(self):
        """Definitioin of needed variables
        
        This function defines variables that are going to be used later
        """
        self._network = None
        self._callbacks = []

    def callback_add_tensorboard(self, log_dir='/tmp/autoencoder',
                                 histogram_freq=1, write_graph=True,
                                 batch_freq=None, variables=['loss', 'val_loss']):
        from callbacks.customTensorBoard import customTensorBoard
        if batch_freq == None: # add standard callback
            tensorboard_callback = TensorBoard(log_dir=log_dir,
                                               histogram_freq=histogram_freq,
                                               write_graph=write_graph)
            # remove a posible tensorboard callback from the list
            # there is no point into keeping several callbacks to tensorflow
            # self._callbacks = [callback for callback in self._callbacks if not
            #                 type(callback) == type(tensorboard_callback)]
            self._callbacks.append(tensorboard_callback)
        else: # add custom callback
            tensorboard_callback = customTensorBoard(log_dir=log_dir,
                                               histogram_freq=histogram_freq,
                                               write_graph=write_graph,
                                               batch_freq=batch_freq,
                                               variables=variables)
            # remove a posible tensorboard callback from the list
            # there is no point into keeping several callbacks to tensorflow
            # self._callbacks = [callback for callback in self._callbacks if not
            #     type(callback) == type(tensorboard_callback)]
            self._callbacks.append(tensorboard_callback)

    def callback_add_earlystopping(self, monitor="val_loss", patience=3):
        from keras.callbacks import EarlyStopping
        earlystopping_callback = EarlyStopping(monitor=monitor,
                                               patience=patience, mode='min')
        self._callbacks = [callback for callback in self._callbacks if not
             type(callback) == type(earlystopping_callback)]
        self._callbacks.append(earlystopping_callback)

    def callback_add_modelcheckpoint(self, filepath, period=1):
        from keras.callbacks import ModelCheckpoint
        modelcheckpoint_callback = ModelCheckpoint(filepath, period=period)
        self._callbacks = [callback for callback in self._callbacks if not
            type(callback) == type(modelcheckpoint_callback)]
        self._callbacks.append(modelcheckpoint_callback)

    def fit_generator(self, batch_size=1000, epochs=1, step=None,
                      train_section=None, val_section=None, history_file=None):
        if step == None: step = self._input_dimension[0]
        if val_section:
            val_generator = SimpleAutoencoderGenerator(self._hdf5_file,
                                                       self._hdf5_group)
            val_generator.configure(sample_length=self._input_dimension[0],
                                    batch_size=batch_size,
                                    step=step,
                                    section=val_section)
            val_data = val_generator.generate_samples(randomise=False)
            val_steps = val_generator.get_nbatches_in_epoch()
        else:
            val_data = None
            val_steps = None

        callbacks = self._callbacks if not self._callbacks == [] else None
        print("Fit generator, train section", train_section)
        print("validation_steps", val_steps)
        self._generator.configure(sample_length=self._input_dimension[0],
                                   batch_size=batch_size,
                                   step=step,
                                   section=train_section)
        print(self._generator.get_nbatches_in_epoch())
        history = self._autoencoder.fit_generator(
            self._generator.generate_samples(),
            self._generator.get_nbatches_in_epoch(),
            validation_data=val_data,
            validation_steps=val_steps,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        if history_file is not None:
            pickle(history.history, history_file)

    def train_dataset(self, batch_size=100, epochs=1, step=None,
                      validation=True):
        """Trains the dataset using the train section

        Gets the train section from the dataset and trains it with that section

        Args:
            batch_size (int): size of the batch
            epochs (int): number of epochs
            step (int): offset between samples
        """
        if step == None: step = self._input_dimension[0]

        # get info of the training section
        data = self._generator.h5g["data"]
        train_section = data.attrs["train_set"]
        train_section = tuple(train_section)
        print("Train section", train_section)
        train_section = self.song_section_to_chunk_section(data, train_section)
        print("Train section", train_section)

        if validation:
            val_section = data.attrs["val_set"]
            val_section = tuple(val_section)
            print("Val section", val_section)
            val_section = self.song_section_to_chunk_section(data, val_section)
            print("Val section", val_section)
        else:
            val_section = None

        # train
        self.fit_generator(batch_size=batch_size, epochs=epochs, step=step,
                           train_section=train_section,
                           val_section=val_section)

    def test_dataset(self):
        pass





class DoubleAutoencoderGenerator(AutoencoderSkeleton):
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
        super().__init__()
        self._input_file = input_file
        self._input_group = input_group
        self._output_file = output_file
        self._output_group = output_group
        self._input_dimension = input_dimension
        self._encoding_dim = encoding_dim
        #FIXME, check next if, I think it' always input_dimension[0]
        if len(input_dimension) >= 2:
            self._sample_length = input_dimension[0]
        elif len(input_dimension) == 1:
            self._sample_length = input_dimension[0]

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

        # use of new DatasetSection
        g_input = input_dataset.group(input_group)
        g_output = output_dataset.group(output_group)
        input_datasection = g_input.get_section(section, stereo=False,
                                                    channel=0)
        output_datasection = g_output.get_section(section, stereo=False,
                                                      channel=0)

        print("---section", section, len(output_datasection))
        # create readers
        input_reader = Reader(input_datasection, sample_length,
                              g_input.chunk_size,
                              step=step, section=section)
        output_reader = Reader(output_datasection, sample_length,
                               g_output.chunk_size,
                               step=step, section=section)

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
        self._network = SimpleAutoencoderNetwork(self._input_dimension,
                                                 self._encoding_dim,
                                                 activation, optimizer, loss)


    def fit_generator(self, batch_size=100, epochs=1, step=None,
                      train_section=None, val_section=None,
                      history_file=None):
        """
        
        Args:
            batch_size: 
            epochs: 
            step: 
            train_section: 
            val_section: provided in songs format
            history_file: 

        Returns:

        """
        if step == None: step = self._input_dimension[0]
        if val_section:
            val_generator = self.create_generator(self._input_file,
                                                  self._input_group,
                                                  self._output_file,
                                                  self._output_group,
                                                  sample_length=self._sample_length,
                                                  step=step,
                                                  section=val_section)

            val_data = val_generator.generate_batches(batch_size=batch_size,
                                                      randomise=False)
            val_steps = val_generator.get_nbatches_in_epoch(batch_size)
        else:
            val_data = None
            val_steps = None

        callbacks = self._callbacks if not self._callbacks == [] else None
        print("Fit generator, train section", train_section)
        print("validation_steps", val_steps)
        train_generator = self.create_generator(self._input_file,
                                           self._input_group,
                                           self._output_file,
                                           self._output_group,
                                           sample_length=self._sample_length,
                                           step=step,
                                           section=train_section)
        train_data = train_generator.generate_batches(batch_size=batch_size,
                                                      randomise=True)

        train_steps = train_generator.get_nbatches_in_epoch(batch_size)
        print(train_generator.get_nbatches_in_epoch(batch_size))
        for layer in self._network.layers:
            print(layer)
        history = self._network.fit_generator(
            train_data,
            train_steps,
            validation_data=val_data,
            validation_steps=val_steps,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
            )
        if history_file is not None:
            pickle(history.history, history_file)


    def train_dataset(self, batch_size=100, epochs=1, step=None,
                      validation=True, history_file=None):
        """Trains the dataset using the train section

        Gets the train section from the dataset and trains it with that section

        Args:
            batch_size (int): size of the batch
            epochs (int): number of epochs
            step (int): offset between samples
        """
        if step == None: step = self._input_dimension[0]

        # get info of the training section
        data = Dataset(self._input_file).group(self._input_group).data()
        train_section = data.attrs["train_set"]
        train_section = tuple(train_section)
        print("Train section (in songs format)", train_section)

        # get info about the validation section
        if validation:
            val_section = data.attrs["val_set"]
            val_section = tuple(val_section)
            print("Val section (in songs format)", val_section)
        else:
            val_section = None

        # train
        self.fit_generator(batch_size=batch_size, epochs=epochs, step=step,
                           train_section=train_section,
                           val_section=val_section, history_file=history_file)