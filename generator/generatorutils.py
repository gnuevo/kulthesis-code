"""Utility functions related to the use of generators

"""

import numpy as np
import math


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


def read_samples(dataset, sample_length, chunk_size, step=None, section=None,
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

    Returns:
        a generator that generates sample after sample following the 
        specifications until the end of the dataset is reached.
    """
    if section == None or not type(section) == tuple:
        section = (0, len(dataset))
    start_of_section = section[0]
    end_of_section = section[1]

    # by the extra substraction we protect ourselves from overflowing the
    # buffer
    #TODO this may not be the best way, but if there are remaining samples
    # from the previoius iteration and we load to memory more from disk we
    # may overflow the buffer
    chunks_in_buffer = buffer_size // chunk_size - chunk_size // \
                                                   sample_length - 1
    if step == None: step=sample_length

    buffer = dataset[start_of_section] # load first chunk
    bottom_index = start_of_section + 1 # +1 because the first chunk (
    # start_of_section) was read already
    top_index = bottom_index # just some initialisation

    step_index = 0 # index to move step by step

    # loop
    while top_index < end_of_section:
        if bottom_index + chunks_in_buffer < end_of_section:
            top_index = bottom_index + chunks_in_buffer
        else:
            top_index = end_of_section

        new_read_data = dataset[bottom_index:top_index]
        bottom_index = top_index
        new_read_data = np.reshape(new_read_data, flatten_shape(new_read_data))
        buffer = np.concatenate((buffer[step_index:], new_read_data), axis=0)
        step_index = 0

        for index in range(0, buffer.shape[0] - sample_length, step):
            # iterate over all the samples taken from the buffer
            yield buffer[index:index+sample_length]

        # prepare old samples to disposal, by doing this, all the samples
        # 'begind' have been used already
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


def total_batches(dataset, sample_length, step, batch_size, section=None,
                  input_format="chunks"):
    """Calculates the total batches in an epoch

    In's intended to use as the steps_per_epoch argument for the 
    fit_generator() function
    
    Equivalent code for this could be done using the range function. For 
    example, for the "songs" case
        math.ceil(len(range(0, total_audio_samples-step+1, step)) / batch_size)
        
    Args:
        dataset (hdf5 dataset): the dataset with samples
        sample_length (int): length of the sample
        step (step): step between samples
        batch_size (int): size of the batches
        input_format (str): "chunks" or "songs", refers to the format of the
            section itself, if the numbers refer to the index of the chunk 
            or the index of the song

    Returns:
        (int), the number of batches
    """
    print("dataset shape", dataset.shape)
    print("section", section)
    if section == None:
        total_audio_samples = dataset.shape[0] * dataset.shape[1]
    else:
        if input_format == "songs":
            songs_lengths = dataset.attrs["songs_lengths"]
            total_audio_samples = sum(songs_lengths[section[0]:section[1]+1
                                  ]) * dataset.shape[1]
        elif input_format == "chunks":
            total_audio_samples = (section[1] - section[0]) * dataset.shape[1]
        else:
            raise ValueError("input_format must be either 'chunks' or "
                             "'songs', gotten", input_format)
    print("Total audio samples = ", total_audio_samples)
    print("Numero de muestras", (total_audio_samples - sample_length) // step)
    n_batches = math.ceil(((total_audio_samples - sample_length) // step) / \
                batch_size)
    print("Total number of batches", n_batches)
    return n_batches


def song_section_to_chunk_section(data, song_section):
    """Converts a song section into a chunck section

    Sections represent portions of the dataset in a tuple format
        (start of section, end of section)
        (start index, end index)
    but the indexes themselves can represent a song or just a chunk of the 
    dataset. This fuction gets the former and converts into the latter, 
    which is the format needed for training.

    Args:
        data: dataset from which metadata will be retreived
        song_section: tuple (start index, end index) 

    Returns:
        (section chunk start index, section chunk end index)

    """
    songs_lengths = [int(length) for length in data.attrs["songs_lengths"]]
    start_index = sum(songs_lengths[0:song_section[0]])
    end_index = sum(songs_lengths[0:song_section[1] + 1])
    return (start_index, end_index)


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