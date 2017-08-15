"""Simple generator for an autoencoder

This is the first generator programmed. It's very simple. In a nutshell it's
just a wrapper over some functions in generatorutils
"""
import h5py
import json
import numpy as np
from .generatorutils import read_samples, total_batches


class SimpleAutoencoderGenerator(object):
    """Reads data from hdf5, formats it and augmentates it
    
    Examples:
        The use of this generator is at follows. In order to have it working
        you need to first instantiate it with a reference to the hdf5 file and 
        group bearing the dataset and a size of the buffer. Then you have to
        configure it and you can finally use it to generate batches of samples.
        Please notice that some of the configurations are mandatory if you 
        want to use the generator. It is easy to reconfigure the generator 
        to generate samples with other characteristics.
        
            # initialise
            generator = SimpleAutoencoderGenerator("myhdf5file", "mygroup", 
                                                        buffer_size=2000000)
            # configure
            generator.configure(sample_length=1000, step=500)
            generator.configure(batch_size=200)
            generator.configure(section=(0,3439))
            # now samples are generated in batches of 200, from the first 
            # 3440 songs in the dataset; they have 1000 sound samples of 
            # size and with an offset of 500 sound samples between them
            
            # now we can generate batches of samples
            # the process is an infinite loop
            for batch in generator.generate_samples():
                print(batch.shape)
                # size should be (200, 1000, 2), the last 2 is because the 
                # songs have 2 audio channels
    """

    def __init__(self, hdf5_file, hdf5_group, buffer_size=1000000):
        """Initialises the generator
        
        Args:
            hdf5_file (str): name of the hdf5 file 
            hdf5_group (str): name of the group inside of the hdf5 file
            buffer_size (int): size (in audio samples) of the buffer used to 
                read parts of the dataset to memory. It is not a rigid limit, 
                therefore may be more audio samples in memory than buffer_size.
                It is just a reference of the number of audio samples that 
                should be in memory in average. A big value of buffer_size 
                prevents from doing to much reads from disk; but if it's too 
                big it could influence the rest of the processing or full 
                the memory.
        """
        self.h5f = h5py.File(hdf5_file, 'r')
        self.h5g = self.h5f[hdf5_group]
        self.buffer_size = buffer_size

        # default configuration
        self._configuration_keys = {"sample_length", "step", "batch_size", "section"}
        self._configuration = {
            "sample_length": None,
            "step": None,
            "batch_size": 1000,
            "section": None
        }

    def configure(self, **kwargs):
        """Sets the configuration
        
        The possible keys are
        sample_length (int): length of the sample (in sound samples).
        step (int): step of the window
        batch_size (int): size of the batch; i.e. how many samples are 
            processed at one
        section (int, int): tuple indicating first and last index of the 
            part of the dataset in which the samples are extracted
            
        Examples:
            This method can be called in the named attribute fashion of 
            Python, providing one or more key-value pairs at once
            
                configure(sample_length=1000)
                configure(step=500)
                # is equivalent to
                configure(sample_length=1000, step=500)
        
        Args:
            **kwargs: dictionary of key-value pairs. Check allowed keys above

        """
        for key in kwargs:
            if key in self._configuration_keys:
                self._configuration[key] = kwargs[key]
            else:
                raise KeyError("Key '{}' is not allowed. It should be in {"
                               "}".format(key, self._configuration_keys))

    def _check_configuration(self):
        """Ensures the configuration is correct, raises error otherwise
        
        Some of the fields could be set to default values by automatically 
        checking other values
        """
        if self._configuration["sample_length"] == None:
            raise ValueError("Value for 'sample_length' is None")
        if self._configuration["step"] == None:
            self._configuration["step"] = self._configuration["sample_length"]
        if self._configuration["batch_size"] == None:
            raise ValueError("Value for 'batch_size' is None")
        if self._configuration["section"] == None:
            try:
                data = self.h5g['data']  # take dataset
            except:
                print("Error getting data to check for configuration "
                      "consistency")
                raise
            self._configuration["section"] = (0, len(data) - 1)

        for key in self._configuration:
            if not key in self._configuration_keys:
                raise KeyError("Key {} in configuration not in configuration"
                               " keys: {}".format(key, self._configuration_keys))

    def generate_samples(self):
        """Generate batches of samples indefinitely
        
        Generates batches of samples indefinitely according to the 
        specifications in the configuration parameters
        
        Returns: a python generator that can be used to generate batch after
            batch

        """
        self._check_configuration()
        try:
            data = self.h5g['data']  # take dataset
            metadata = json.loads(data.attrs['metadata'])
            chunk_size = metadata['chunk_size']
            nsongs = metadata['Nsongs']
        except:
            print("Error getting data. Some of the needed fields do not exist")
            raise

        # read variables from configuration
        sample_length = self._configuration["sample_length"]
        step = self._configuration["step"]
        batch_size = self._configuration["batch_size"]
        section = self._configuration["section"]

        # create array to store the batch
        batch_features = np.zeros((batch_size, sample_length, 2)) # FIXME, hadcoded 2, because there are 2 audio channels
        gen = read_samples(data, sample_length, chunk_size,
                                                step=step,
                                                buffer_size=self.buffer_size)
        while True: # generators are intended to work indefinitely
            for index in range(batch_size):
                try:
                    sample = next(gen)
                except StopIteration:
                    # restart the generator
                    gen = read_samples(data, sample_length, chunk_size,
                                       step=step,
                                       buffer_size=self.buffer_size,
                                       section=section)
                    sample = next(gen)
                batch_features[index] = sample
            yield batch_features, batch_features

    def get_nbatches_in_epoch(self):
        """Returns the number of batches that make an epoch
        
        The number of batches that make an epoch changes with the 
        configuration. For example, the step parameter allows the 
        augmentation of data. This method calculates and returns that number
        
        Returns: (int) number of batches that make an epoch

        """
        self._check_configuration()
        return total_batches(self.h5g['data'],
                             self._configuration["sample_length"],
                             self._configuration["step"],
                             self._configuration["batch_size"],
                             section=self._configuration["section"])