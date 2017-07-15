"""Simple generator for an autoencoder

This is the first generator programmed. It's very simple. In a nutshell it's
just a wrapper over some functions in generatorutils
"""
import h5py
import generatorutils as genutils

class SimpleAutoencoderGenerator(object):
    """Reads data from hdf5, formats it and augmentates it
    
    """

    def __init__(self, hdf5_file, hdf5_group, buffer_size=1000000):
        self.h5f = h5py.File(hdf5_file, 'r')
        self.h5g = self.h5f[hdf5_group]
        self.buffer_size = buffer_size

    def generate_samples(self, sample_length, step=None):
        if step == None: step = sample_length
        try:
            data = self.h5g['data']  # take dataset
            metadata = self.h5g.attrs['metadata']
            chunk_size = metadata['chunk_size']
            nsongs = metadata['Nsongs']
        except:
            print("Error getting data. Some of the needed fields do not exist")
            raise
        for sample in genutils.read_samples(data, sample_length, chunk_size,
                                            step=step,
                                            buffer_size=self.buffer_size):
            yield sample, sample