"""Utility functions related to the use of generators

"""

import numpy as np


BUFFER_SIZE = 100000 # size in samples of the

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


def read_samples(dataset, sample_length, chunk_size, step=None,
                 buffer_size=BUFFER_SIZE):
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
    dataset may be under ignored.
    
    Args:
        dataset: hdf5 dataset or np.array containing the audio samples. It 
        is assumed that it is divided into chunks
        sample_length (int): desired output length of the samples
        chunk_size (int): size of the chunks in the dataset
        step (int): (optional) step between samples 
        buffer_size (int): (optional) 

    Returns:
        a generator that generates sample after sample following the 
        specifications until the end of the dataset is reached.
    """
    chunks_in_buffer = buffer_size // chunk_size
    dataset_length = len(dataset)
    if step == None: step=sample_length

    buffer = dataset[0] #load first chunk
    bottom_index = 1 # 1 because the first chunk (0) was read already
    top_index = bottom_index # just some initialisation

    step_index = 0 # index to move step by step

    # loop
    while top_index < dataset_length:
        # TODO last audio samples may not be used due to the break condition
        if bottom_index+chunks_in_buffer < dataset_length:
            top_index = bottom_index+chunks_in_buffer
        else:
            top_index = dataset_length

        new_read_data = dataset[bottom_index:top_index]
        new_read_data = np.reshape(new_read_data, flatten_shape(new_read_data))
        buffer = np.concatenate((buffer[step_index:], new_read_data), axis=0)
        step_index = 0

        for index in range(0,buffer.shape[0] - sample_length, step):
            # iterate over all the samples taken from the buffer
            yield buffer[index:index+sample_length]

        # prepare old samples to disposal
        step_index = index + step


def read_group_data(hdf5_file, group_name, sample_length, step=None):
    # FIXME function not used
    if step == None: step = sample_length
    try:
        group = hdf5_file[group_name]
        data = group['data']  # take dataset
        metadata = group.attrs['metadata']
        chunk_size = metadata['chunk_size']
        nsongs = metadata['Nsongs']
    except:
        print("Error getting data. Some of the needed fields do not exist")
        raise

    for sample in read_samples(data, sample_length, chunk_size, step,
                               BUFFER_SIZE):
        yield sample


def total_batches(dataset, sample_length, step, batch_size):
    """Calculates the total batches in an epoch

    In's intended to use as the steps_per_epoch argument for the 
    fit_generator() function
        
    Args:
        dataset (hdf5 dataset): the dataset with samples
        sample_length (int): length of the sample
        step (step): step between samples
        batch_size (int): size of the batches

    Returns:

    """
    total_audio_samples = dataset.shape[0] * dataset.shape[1]
    return ((total_audio_samples - sample_length) // step) // batch_size

if __name__ == "__main__":
    print("Main")
    import h5py
    f = h5py.File('/home/grego/MAI/MasterThesis/midiutil/dataset/v2_dataset'
                  '.hdf5','r')
    d = f['/timbre0/data']

    count = 0
    for sample in read_samples(d, 500, 5000, step=50, buffer_size=50000):
        count += 1
        if count == 1000000: break
    print(count)