"""This file contains batchers

A batcher is the immediate upper level of the reader. It takes samples from
the reader and converts them into batches of samples. Extra functions like 
randomisation may also be applied.
"""
from random import shuffle
import numpy as np
from .readers import Function
from generator.generatorutils import total_batches


class EndOfEpoch(Exception):
    """This exception depict the batcher has successfully finished 
    processing the entire dataset, i.e. the epoch is complete
    
    """
    def __init__(self, message=""):
        self.message = message


class DoubleSynchronisedRandomisedBatcher(object):
    """Batcher that creates batches for both input and output at a time
    
    It is randomised, so the samples can be randomised at will. The 
    randomisation of the samples is kept the same for input and output, 
    that is why synchronised.
    
    """
    def __init__(self, input_reader, output_reader, buffer_size=1000):
        """Initialised the batcher
        
        Args:
            input_reader: reader for input samples
            output_reader: reader for output samples
            buffer_size: size of the randomisation buffer, in number of samples
        """
        self.input_reader = input_reader
        self.output_reader = output_reader
        self.buffer_size = buffer_size
        if not input_reader.sample_length == output_reader.sample_length:
            raise ValueError("The sample_length for the readers differ")
        else:
            self.sample_length = input_reader.sample_length

    def get_nbatches_in_epoch(self, batch_size):
        """Returns the number of batches that make an epoch

        The number of batches that make an epoch changes with the 
        configuration. For example, the step parameter allows the 
        augmentation of data. This method calculates and returns that number

        Returns: (int) number of batches that make an epoch

        """
        print(type(self.input_reader.dataset))
        dataset = self.input_reader.dataset
        sample_length = self.input_reader.sample_length
        step = self.input_reader.step
        section = self.input_reader.get_configuration()["reader:section"]
        # readers always get their section in the form of 'songs'
        input_format = "songs"
        return total_batches(dataset, sample_length, step, batch_size,
                             section=section, input_format=input_format)

    def _divide_in_batches(self, samples, num_samples, batch_size):
        """Organises a list of samples into a list of batches
        
        The last batch may not be complete if there are not enough samples 
        left.
        
        Args:
            samples: iterable of samples
            num_samples: total number of samples in the iterable
            batch_size: size of the batch

        Returns:

        """
        for index in range(0, num_samples, batch_size):
            upper_index = min(index+batch_size, num_samples)
            yield samples[index:upper_index]

    def generate_batches(self, batch_size=100, randomise=True,
                         function=None, function_args=[]):
        """Generates batches indefinitely
        
        Args:
            batch_size: 
            randomise: 
            function: 
            function_args: 

        Returns:

        """
        if function is not None:
            function = Function(function, *function_args)
        # calculate the number of batches that fit in the buffer
        batches_in_buffer = self.buffer_size // batch_size
        # create array to store the batch
        print("SIZE OF THE BUFFER")
        print((batches_in_buffer * batch_size, \
                                    self.sample_length, 2))
        buffer_size = batches_in_buffer * batch_size
        sample_shape = self.input_reader.sample_shape()
        print("sample shape", sample_shape)
        print("type buffer size", type(buffer_size), buffer_size)
        buffer_shape = [buffer_size]
        print("")
        buffer_shape.extend(list(sample_shape))
        buffer_shape = tuple(buffer_shape)
        print("buffer shape", buffer_shape)
        input_buffer_features = np.zeros(buffer_shape)
        output_buffer_features = np.zeros(buffer_shape)

        input_gen = self.input_reader.read_samples(function=function)
        output_gen = self.output_reader.read_samples(function=function)
        while True:
            for index in range(batches_in_buffer * batch_size):
                try:
                    input_sample = next(input_gen)
                    output_sample = next(output_gen)
                except StopIteration:
                    # raise exception in case this would be useful
                    # raise EndOfEpoch()
                    input_gen = self.input_reader.read_samples(function=function)
                    output_gen = self.output_reader.read_samples(function=function)
                    # when buffers are full, break and return samples
                    break
                # load samples in the buffer
                input_buffer_features[index] = input_sample
                output_buffer_features[index] = output_sample

            # randomise if needed
            if randomise:
                # generate random indices
                indices = list(range(index + 1))
                shuffle(indices)
                # randomise the batch
                input_randomised_buffer = input_buffer_features[indices]
                output_randomised_buffer = output_buffer_features[indices]
            else:
                input_randomised_buffer = input_buffer_features
                output_randomised_buffer = output_buffer_features

            # return batches
            for input_batch, output_batch in zip(
                    self._divide_in_batches(input_randomised_buffer,
                                            index+1, batch_size),
                    self._divide_in_batches(output_randomised_buffer,
                                            index+1,batch_size)):
                yield input_batch, output_batch