"""This file is intended to store readers

Readers are code that read samples from the dataset
"""
import h5py
import numpy as np
from generator.generatorutils import song_section_to_chunk_section


def flatten_shape(array):
    """Returns the new shape of the array if the first dimension is flatten

    This is basically a method that returns the shape of the array with the 
    two first dimensions multiplied (and thus merged as the new first 
    dimension)

    Args:
        array (np.ndarray): array to be flatten

    Returns (tuple): new shape
    """
    if len(array.shape) < 2:
        raise ValueError("array variable must have at least 2 dimensions")
    new_dim = array.shape[0] * array.shape[1]
    if len(array.shape) == 2:
        new_shape = (new_dim,)
    elif len(array.shape) > 2:
        new_shape = [new_dim]
        new_shape.extend(array.shape[2:])
        new_shape = tuple(new_shape)
    return new_shape


def Dataset(file, group):
    """Takes file and group and returns the corresponding dataset
    
    Args:
        file: name of the hdf5 file
        group: group inside the hdf5 file

    Returns: the dataset

    """
    h5f = h5py.File(file, 'r')
    h5g = h5f[group]
    dataset = h5g['data']
    return dataset


def Function(f_to_bind, *args):
    """Binds to a custom function
    
    Similar behaviour to functools.partial. The idea is to froze the *args 
    section from f_to_bind. f_to_bind have to be of the form
    
        f_to_bind(x, *args)
    
    where x is a sample, and *args are the additional parameter it may need.
    For example if we want to add the same number to all our samples we can do
    
        add_number(x, number):
            return x + number
        
        # imagine we want to add 2
        f = Function(add_number, 2)
        
        # now you can easily modify the samples by doing
        new_sample = f(sample)
    
    Args:
        f_to_bind: function of the form
        *args: extra arguments f_to_bind may need

    Returns: bound function

    """
    def mock_function(x):
        return f_to_bind(x, *args)
    return mock_function


class Reader(object):
    """Creates the samples reading the input dataset (np.array)

        This procedure includes online processing (chunks of the dataset are 
        read while execution and they are formated into the desired length); and 
        data augmentation (using the step argument we can augmentate the number 
        of samples generated).

        It is worth to notice these characteristics:
        1. The dataset is processed as a whole. No distinction between songs is 
        made. Therefore samples from one song can be mixed with samples from the
        previous or next song.
        2. The step procedure doesn't add artificial 0's at the end in case that is
        necessary to complete a sample. Therefore, the last audio samples of the
        dataset may be ignored.
        """
    def __init__(self, dataset, sample_length, chunk_size, step=None,
                 section=None, buffer_size=1000000, left_padding=0):
        """Initialises the object and sets the necessary values
        
        Args:
            dataset: hdf5 dataset or np.array containing the audio samples. It 
            is assumed that it is divided into chunks
            sample_length (int): desired output length of the samples
            chunk_size (int): size of the chunks in the dataset
            section (tuple): indicates the section intended to read inside the 
                dataset. E.g. if section=(0,100) then the function reads 
                elements from 0 to 100 of the dataset (both included). If None 
                then all the dataset is read.
            step (int): (optional) step between samples 
            buffer_size (int): (optional) number of samples that are read to 
                memory at once
            section (int, int): (tuple) first and last index of the section of 
                the dataset that is intended to read. The dataset is virtually 
                limited to the section from first index to last index
            left_padding (int): sets the number of 0 amplitude audio samples 
                added by default when generating the training samples. This 
                helps to generate the test starting from different moments.
        """
        self.dataset = dataset
        self.sample_length = sample_length
        self.chunk_size = chunk_size
        self.step = step
        #FIXME, now no section is provided, the same behaviour is achieved
        # using DataSection
        # self.section = song_section_to_chunk_section(dataset, section)
        self.section = None
        self.buffer_size = buffer_size
        self.left_padding = left_padding

        self.__configuration = {
            "reader:dataset": dataset,
            "reader:sample_length": sample_length,
            "reader:chunk_size": chunk_size,
            "reader:step": step,
            "reader:section": section,
            "reader:buffer_size": buffer_size
        }

    def get_configuration(self):
        return self.__configuration

    def sample_shape(self):
        """Returns the shape of the sample
        
        Basically helps to distinguish samples that are stereo from samples 
        that are mono
        """
        shape = [self.sample_length]
        shape.extend(list(self.dataset.shape[2:]))
        return tuple(shape)

    def read_samples(self, function=None):
        """Generates samples following the specifications

            Args:
                function: the function to apply to every sample. Use Function()
                
            Returns:
                a generator that generates sample after sample following the 
                specifications until the end of the dataset is reached.
        """
        # extract object variables to local variables
        if self.section == None or not type(self.section) == tuple:
            section = (0, len(self.dataset))
        else:
            section = self.section
        start_of_section = section[0]
        end_of_section = section[1]

        dataset = self.dataset
        buffer_size = self.buffer_size
        chunk_size = self.chunk_size
        sample_length = self.sample_length
        step = self.step
        if step == None: step = sample_length

        # by the extra substraction we protect ourselves from overflowing the
        # buffer
        # TODO this may not be the best way, but if there are remaining samples
        # from the previoius iteration and we load to memory more from disk we
        # may overflow the buffer
        chunks_in_buffer = buffer_size // chunk_size - chunk_size // \
                                                       sample_length - 1

        if self.left_padding >= 0:
            buffer = np.zeros(self.left_padding)
            bottom_index = 0
        else:
            # FIXME, the new modifications don't use this part of the code
            # buffer = dataset[start_of_section]  # load first chunk
            # bottom_index = start_of_section + 1  # +1 because the first chunk (
            buffer = np.array([])
            bottom_index = 0
        # start_of_section) was read already
        top_index = bottom_index  # just some initialisation

        step_index = 0  # index to move step by step

        # loop
        while top_index < end_of_section:
            if bottom_index + chunks_in_buffer < end_of_section:
                top_index = bottom_index + chunks_in_buffer
            else:
                top_index = end_of_section

            new_read_data = dataset[bottom_index:top_index]
            bottom_index = top_index
            new_read_data = np.reshape(new_read_data,
                                       flatten_shape(new_read_data))
            buffer = np.concatenate((buffer[step_index:], new_read_data),
                                    axis=0)
            step_index = 0
            for index in range(0, buffer.shape[0] - sample_length, step):
                # iterate over all the samples taken from the buffer
                sample = buffer[index:index + sample_length]
                if function is not None:
                    sample = function(sample)
                yield sample

            # prepare old samples to disposal, by doing this, all the samples
            # 'begind' have been used already
            step_index = index + step