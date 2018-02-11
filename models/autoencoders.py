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
from .networks import SimpleAutoencoderNetwork, DeepAutoencoderNetwork
import tensorflow as tf
import numpy as np
import scipy.io.wavfile as wavfile
import json
from keras.models import load_model
from abc import abstractmethod

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

    def callback_add_custommodelcheckpoint(self, dirpath, period=1):
        from callbacks.customCheckPoint import customModelCheckpoint
        callback = customModelCheckpoint(self, dirpath, period=period)
        self._callbacks = [cbk for cbk in self._callbacks if not
            type(cbk) == type(callback)]
        self._callbacks.append(callback)

    def callback_learning_rate_scheduler(self, initial_lr, decay):
        from keras.callbacks import LearningRateScheduler
        def lrscheduler(epoch):
            new_lr = initial_lr * 1.0/(1 + decay * epoch)
            return new_lr
        scheduler = LearningRateScheduler(lrscheduler)
        self._callbacks = [callback for callback in self._callbacks if not
            type(callback) == type(scheduler)]
        self._callbacks.append(scheduler)

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
        self._generator.configure(sample_length=self._input_dimension[0],
                                   batch_size=batch_size,
                                   step=step,
                                   section=train_section)
        history = self._autoencoder.fit_generator(
            self._generator.generate_samples(),
            self._generator.get_nbatches_in_epoch(),
            validation_data=val_data,
            validation_steps=val_steps,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
            initial_epoch=self._last_epoch
        )
        self._last_epoch = epochs
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
        train_section = self.song_section_to_chunk_section(data, train_section)

        if validation:
            val_section = data.attrs["val_set"]
            val_section = tuple(val_section)
            val_section = self.song_section_to_chunk_section(data, val_section)
        else:
            val_section = None

        # train
        self.fit_generator(batch_size=batch_size, epochs=epochs, step=step,
                           train_section=train_section,
                           val_section=val_section)

    def test_dataset(self):
        pass

    def _get_config(self):
        return dict()

    def save(self, directory, extra_config={}):
        if not directory[-1] == "/": directory += "/"
        config = self._get_config()
        print(config)
        for key in extra_config.keys():
            config[key] = extra_config[key]
        with open(directory + "model.json", "w") as f:
            f.write(json.dumps(config, indent=4))
            print(json.dumps(config))
        self._network.save(directory + "model.h5", overwrite=True,
                           include_optimizer=True)
        print("Saving model in", directory)

    def _load(self, config):
        pass

    def load(self, directory):
        if not directory[-1] == "/": directory += "/"
        config = json.loads(open(directory + "model.json", "r"))
        self._load(config)
        self._network = load_model(directory + "model.h5")


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
                         step=None, section=None, left_padding=0):
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
        if step == None: step = self._input_dimension[0]

        # get datasets
        input_dataset = Dataset(input_file)
        output_dataset = Dataset(output_file)

        # use of new DatasetSection
        g_input = input_dataset.group(input_group)
        g_output = output_dataset.group(output_group)
        input_datasection = g_input.get_section(section, stereo=False,
                                                    channel=0).standarized()
        output_datasection = g_output.get_section(section, stereo=False,
                                                      channel=0).standarized()

        # create readers
        input_reader = Reader(input_datasection, sample_length,
                              g_input.chunk_size,
                              step=step, section=section,
                              left_padding=left_padding)
        output_reader = Reader(output_datasection, sample_length,
                               g_output.chunk_size,
                               step=step, section=section,
                               left_padding=left_padding)

        # create batcher
        batcher = DSRBatcher(input_reader, output_reader)
        return batcher


    def initialise(self, activation='relu', optimizer='adadelta',
                   loss='binary_crossentropy'):
        """Initialise the neural network

        The neural network gets already initialised during __init__ so if 
        this method is called the network behind is restarted.
        """
        # keep track of the loss in case it is a custom one
        self.loss = loss
        # call to the parent method
        self._network = SimpleAutoencoderNetwork(self._input_dimension,
                                                 self._encoding_dim,
                                                 activation, optimizer, loss)


    def fit_generator(self, batch_size=100, epochs=1, step=None,
                      train_section=None, val_section=None,
                      history_file=None, function=None, function_args=[]):
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
                                                      randomise=False,
                                                      function=function,
                                                      function_args=function_args)
            val_steps = val_generator.get_nbatches_in_epoch(batch_size)
        else:
            val_data = None
            val_steps = None

        callbacks = self._callbacks if not self._callbacks == [] else None
        train_generator = self.create_generator(self._input_file,
                                           self._input_group,
                                           self._output_file,
                                           self._output_group,
                                           sample_length=self._sample_length,
                                           step=step,
                                           section=train_section)
        train_data = train_generator.generate_batches(batch_size=batch_size,
                                                      randomise=True,
                                                      function=function,
                                                      function_args=function_args)

        train_steps = train_generator.get_nbatches_in_epoch(batch_size)
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
                      validation=True, history_file=None, function=None,
                      function_args=[], sample_weights=None):
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

        # get info about the validation section
        if validation:
            val_section = data.attrs["val_set"]
            val_section = tuple(val_section)
        else:
            val_section = None

        # train
        self.fit_generator(batch_size=batch_size, epochs=epochs, step=step,
                           train_section=train_section,
                           val_section=val_section, history_file=history_file,
                           function=function, function_args=function_args)


    def evaluate_generator(self, evaluate_section=None, batch_size=100,
                            step=None, function=None, function_args=[]):
        """
        
        Args:
            evaluate_section: 
            batch_size: 
            step: step to generate samples (do not confuse with `steps` from
                model.evaluate_generator(steps=...)

        Returns:

        """
        if step == None: step = self._input_dimension[0]

        # callbacks = self._callbacks if not self._callbacks == [] else None

        test_generator = self.create_generator(self._input_file,
                                           self._input_group,
                                           self._output_file,
                                           self._output_group,
                                           sample_length=self._sample_length,
                                           step=step,
                                           section=evaluate_section)
        test_data = test_generator.generate_batches(batch_size=batch_size,
                                                      randomise=False,
                                                    function=function,
                                                      function_args=function_args)

        test_steps = test_generator.get_nbatches_in_epoch(batch_size)
        evaluation = self._network.evaluate_generator(
            test_data,
            steps=test_steps
            )
        return evaluation


    def test_dataset(self, batch_size=100, step=None):
        """Tests the dataset
        
        Args:
            batch_size: 
            step: recommended to keep it None

        Returns:

        """
        if step == None: step = self._input_dimension[0]

        # get info on the test section
        data = Dataset(self._input_file).group(self._input_group).data()
        test_section = data.attrs["test_set"]
        test_section = tuple(test_section)

        # test
        return self.evaluate_generator(evaluate_section=test_section,
                                       batch_size=batch_size, step=step)

    def recover_audio(self, batch_size=100, tblogdir="/tmp/autoencoder",
                      function=None, function_args=[], num_average=1,
                      test_out=None, predict_out=None, targets_out=None):
        """Function to recover the audio samples using tensorboard audio 
        summary
        
        Returns:

        """

        # get info on the test section
        data = Dataset(self._input_file).group(self._input_group).data()
        test_section = data.attrs["test_set"]
        song_lengths = data.attrs["songs_lengths"]
        test_section = tuple(test_section)

        predictions = []
        padding = 203
        for i in range(num_average):
            test_generator = self.create_generator(self._input_file,
                                                  self._input_group,
                                                  self._output_file,
                                                  self._output_group,
                                                  sample_length=self._sample_length,
                                                  step=None,
                                                  section=test_section,
                                                  left_padding=padding*i)
            test_data = test_generator.generate_batches(batch_size=batch_size,
                                                        randomise=False, function=function,
                                                          function_args=function_args)
            test_steps = test_generator.get_nbatches_in_epoch(batch_size=batch_size)
            print("test steps", test_steps)

            print("Test steps", test_steps  )
            prediction = self._network.predict_generator(test_data,
                                                         test_steps)
            print(i, "prediction shape before", prediction.shape)
            prediction = np.reshape(prediction, prediction.shape[0] *
                                    prediction.shape[1])
            print(i, "prediction shape after reshape", prediction.shape)
            prediction = prediction[padding*i:]
            # prediction = np.pad(prediction, ((num_average -1 -i)*padding, 0),
            #                     'constant')
            print(i, "prediction shape after", prediction.shape)
            predictions.append(prediction)
        print("type prediction", type(prediction))
        print("shape of prediction", prediction.shape)
        print("Looks good for now!!")

        lengths = [p.shape[0] for p in predictions]
        print("the lengths of the predictions", lengths)
        maximum = max(lengths)
        average = np.zeros(maximum)
        for p in predictions:
            p = np.pad(p, (0,maximum - p.shape[0]), "constant")
            average += p
        average = average / num_average

        test_data = test_generator.generate_batches(batch_size=batch_size,
                                                    randomise=False, function=function,
                                                      function_args=function_args)

        def generator_filter():
            for elem in test_data:
                yield elem[0]

        t_data = np.ndarray([])
        for _ in range(test_steps):
            new_data = next(generator_filter())
            t_data = np.append(t_data, new_data)
        print("shape t_data", t_data.shape)
        # working with tensorboard stuff
        tb_writer = tf.summary.FileWriter(tblogdir)
        p_data = average
        print("p_data shape", p_data.shape)
        # flattened_shape = t_data.shape[0] * t_data.shape[1]
        flattened_shape = t_data.shape[0]
        print("flattened shape", flattened_shape)

        path = "/home/grego/PycharmProjects/style-transfer/output/"
        name = "average{}_e{}_{}_{}_".format(num_average, self._last_epoch,
                                             '-'.join(str(
            self._middle_layers)), self._encoding_dim)
        if not test_out == None:
            wavfile.write(test_out + "_test_{}.wav".format(name), 22050, t_data)
        t_data = np.reshape(t_data, flattened_shape)
        if not predict_out == None:
            wavfile.write(predict_out + "_predicted_{}.wav".format(name), 22050, p_data)

    def _get_config(self):
        super_config = super()._get_config()
        local_config = {
            "input_file": self._input_file,
            "input_group": self._input_group,
            "output_file": self._output_file,
            "output_group": self._output_group,
            "input_dimension": self._input_dimension,
            "encoding_dim": self._encoding_dim,
            "sample_length": self._sample_length,
            "class": type(self)
        }
        if not type(self.loss) == str:
            print("Saving, loss", self.loss)
            # custom loss
            local_config["loss"] = self.loss.get_code()
        else:
            local_config["loss"] = self.loss
        config = {**super_config, **local_config}
        return config

    def _load(self, config):
        super()._load(config)
        self._input_file = config["input_file"]
        self._input_group = config["input_group"]
        self._output_file = config["output_file"]
        self._output_group = config["output_group"]
        self._input_dimension = config["input_dimension"]
        self._encoding_dim = config["encoding_dim"]
        self._sample_length = config["sample_lenght"]


class DeepDoubleAutoencoderGenerator(DoubleAutoencoderGenerator):

    def __init__(self, input_file, input_group, output_file,
                 output_group, input_dimension=(1000, 2), encoding_dim=32,
                 middle_layers=[]):
        super().__init__(input_file, input_group, output_file, output_group,\
        input_dimension=input_dimension, encoding_dim=encoding_dim)
        self._middle_layers = middle_layers

    def initialise(self, activation='relu', optimizer='adadelta',
                   loss='binary_crossentropy'):
        # keep track of the loss in case it is a custom one
        self.loss = loss
        self._last_epoch = 0
        self._network = DeepAutoencoderNetwork(self._input_dimension,
                                               self._middle_layers,
                                               self._encoding_dim,
                                               activation, optimizer, loss)

    def _get_config(self):
        config = super()._get_config()
        extra_config = {
            "middle_layers": self._middle_layers,
            "class": type(self).__name__
        }
        for key in extra_config.keys():
            config[key] = extra_config[key]
        return config

    @abstractmethod
    def load(directory):
        print("I am f*cking loading!!!!!!!!!!!!!!")
        if not directory[-1] == "/": directory += "/"
        print("open", directory + "model.json")
        config = json.load(open(directory + "model.json", "r"))
        model = DeepDoubleAutoencoderGenerator(config["input_file"],
                                               config["input_group"],
                                               config["output_file"],
                                               config["output_group"],
                                               config["input_dimension"], 
                                               config["encoding_dim"], 
                                               config["middle_layers"])
        model._last_epoch = config["last_epoch"]
        # import keras.losses
        # keras.losses.custom_loss = custom_loss
        custom_objects = dict()
        if "loss" in config:
            # load custom loss
            from .losses import CustomLoss, WeightedMSE
            loss = CustomLoss.code_to_loss(config["loss"])
            print("culo")
            if not type(loss) == str:
                custom_objects = {loss.__name__: loss}
            print("custom objects", custom_objects)
        model._network = load_model(directory + "model.h5", custom_objects=custom_objects)
        return model