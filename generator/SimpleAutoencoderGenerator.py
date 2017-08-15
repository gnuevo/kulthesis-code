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
    
    """

    def __init__(self, hdf5_file, hdf5_group, buffer_size=1000000):
        self.h5f = h5py.File(hdf5_file, 'r')
        self.h5g = self.h5f[hdf5_group]
        self.buffer_size = buffer_size

        # default configuration
        self.__configuration = {
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
            self.__configuration[key] = kwargs[key]

    def generate_samples(self, sample_length, step=None, batch_size=1000,
                         section=None):
        section = (0, 3439)
        if step == None: step = sample_length
        try:
            data = self.h5g['data']  # take dataset
            metadata = json.loads(data.attrs['metadata'])
            chunk_size = metadata['chunk_size']
            nsongs = metadata['Nsongs']
        except:
            print("Error getting data. Some of the needed fields do not exist")
            raise

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

    def get_nsamples(self, sample_length, step=None, batch_size=None,
                     section=None):
        if step == None: step = sample_length
        return total_batches(self.h5g['data'], sample_length, step,
                             batch_size, section=section)