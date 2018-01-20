"""

"""
import h5py
import math
import json
from collections import Sequence
from generator.generatorutils import song_section_to_chunk_section


class DatasetFormatError(Exception):
    def __init__(self, message):
        self.message = message


class Dataset(object):
    def __init__(self, dataset_file, mode='r'):
        self._dataset = h5py.File(dataset_file, mode)

    def group(self, group_name):
        return Group(self._dataset[group_name])

    def total_batches(self, sample_length, step, batch_size, section=None,
                      input_format="chunks"):
        num_batches = []
        for group in self._dataset:
            num_batches.append(self.group(group).total_batches(
                sample_length, step, batch_size, section, input_format))
        set_num_batches = set(num_batches)
        if len(set_num_batches) > 1:
            raise DatasetFormatError("The groups of the dataset contain a "
                                     "different number of batches")


class Group(object):
    def __init__(self, group):
        self._group = group
        metadata = json.loads(group["data"].attrs["metadata"])
        self.chunk_size = metadata['chunk_size']
        self.nsongs = metadata["Nsongs"]

    def data(self):
        return self._group["data"]

    def total_batches(self, sample_length, step, batch_size, section=None,
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
        dataset = self._group.data
        print("dataset shape", dataset.shape)
        print("section", section)
        if section == None:
            total_audio_samples = dataset.shape[0] * dataset.shape[1]
        else:
            if input_format == "songs":
                songs_lengths = dataset.attrs["songs_lengths"]
                total_audio_samples = sum(
                    songs_lengths[section[0]:section[1] + 1
                    ]) * dataset.shape[1]
            elif input_format == "chunks":
                total_audio_samples = (section[1] - section[0]) * \
                                      dataset.shape[1]
            else:
                raise ValueError("input_format must be either 'chunks' or "
                                 "'songs', gotten", input_format)
        print("Total audio samples = ", total_audio_samples)
        print("Numero de muestras",
              (total_audio_samples - sample_length) // step)
        n_batches = math.ceil(((total_audio_samples - sample_length) // step) / \
                              batch_size)
        print("Total number of batches", n_batches)
        return n_batches


class DatasetSection(Sequence):
    """Encapsulates a section of the dataset with a Sequence easy access API
    
    The idea behind this is that the DataSection object reads chunks at a 
    low level, allowing to perform some processing to data that would be 
    considered as preprocessing for the network. For example, selecting only
    one channel of audio, normalising the dataset, etc. Therefore, 
    the Reader only cares about doing processing to the signal related to 
    training, for example adding noise to challenge the autoencoder.
    
    """

    def __init__(self, group_data, section=None, section_format="songs",
                 stereo=False,
                 channel=0):
        """
        
        Args:
            group_data: Group.data()
            section: tuple with the boundaries of the section
            section_format: format of the section ('songs' or 'chunks')
            stereo: if you want to treat data as stereo or mono
            channel: channel you want to get if stereo=False and data is stereo
        """
        self._group_data = group_data
        if section == None:
            self._start_section = 0
            self._stop_section = group_data # FIXME, I think this is
            # wrong, len() - 1?
        else:
            if section_format == "songs":
                chunk_section = song_section_to_chunk_section(group_data,
                                                              song_section=section)
                self._start_section = chunk_section[0]
                self._stop_section = chunk_section[1]
            elif section_format == "chunks":
                self._start_section = section[0]
                self._stop_section = section[1]
            else:
                raise ValueError("section_format must be either 'songs' or "
                                 "'chunks', gotten", section_format)

        # create a filter function to extract the desired channel(s)
        if stereo == True:
            self._filter = lambda x: x
        elif stereo == False:
            if type(channel) == int and channel < self._group_data.shape[-1]:
                # return the selected channel
                self._filter = lambda x: x[:, channel]
            else:
                raise ValueError("channel is not <int> or is out of range, "
                                 "channel=", channel)

    def __len__(self):
        # FIXME, I'm not sure this is the correct function
        return self._stop_section - self._start_section

    def __getitem__(self, item):
        # in order to retrieve data we have to index over the section by
        # adding self._start_section to our indices
        if type(item) == int:
            item += self._start_section
        elif type(item) == slice:
            item.start += self._start_section
            item.stop += self._start_section
        else:
            raise TypeError("Item type is not <int> nor <slice>, type",
                            type(item))
        retrieved = self._group_data[item]
        return self._filter(retrieved)
