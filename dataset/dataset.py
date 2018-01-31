"""

"""
import h5py
import math
import json
from collections import Sequence
from generator.generatorutils import song_section_to_chunk_section
import numpy as np


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
        n_batches = math.ceil(((total_audio_samples - sample_length) // step) / \
                              batch_size)
        return n_batches

    def get_section(self, section, stereo=False, channel=0):
        """Returns a section of the dataset wrapped into a DataSection object
        
        Args:
            section: tuple (start_section, stop_section)

        Returns:

        """
        return DataSection(self._group["data"], section=section,
                           section_format="songs", stereo=stereo, channel=channel)

    def get_train_section(self, stereo=False, channel=0):
        return self._get_section(self.data().attrs["train_set"],
                                 stereo=stereo, channel=channel)

    def get_val_section(self, stereo=False, channel=0):
        return self._get_section(self.data().attrs["val_set"],
                                 stereo=stereo, channel=channel)

    def get_test_section(self, stereo=False, channel=0):
        return self._get_section(self.data().attrs["test_set"],
                                 stereo=stereo, channel=channel)


class DataSection(Sequence):
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
        self.group_data = group_data
        self.chunk_size = self.group_data.shape[1]
        self.attrs = self.group_data.attrs
        if section == None:
            self._start_section = 0
            self._stop_section = len(group_data)
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
        self.shape = list(self.group_data.shape)
        self.shape[0] = self._stop_section - self._start_section + 1
        self.shape = tuple(self.shape)
        print("datasection, start, stop, shape", self._start_section,
              self._stop_section, self.shape)

        # create a filter function to extract the desired channel(s)
        if stereo == True:
            self._channel_filter = slice(None, None, None)
            self.shape = self.shape
        elif stereo == False:
            if type(channel) == int and channel < self.group_data.shape[-1]:
                # return the selected channel
                self._channel_filter = channel
                self.shape = self.shape[:-1] # ignore last
                print("herrrr")
            else:
                raise ValueError("channel is not <int> or is out of range, "
                                 "channel=", channel)
        print("la estoy liando ?datasection, start, stop, shape",
              self._start_section,
              self._stop_section, self.shape)
        print("")

    def __len__(self):
        return self._stop_section - self._start_section

    def __getitem__(self, item):
        # in order to retrieve data we have to index over the section by
        # adding self._start_section to our indices
        if type(item) == int:
            item += self._start_section
        elif type(item) == slice:
            start = item.start + self._start_section
            stop = item.stop + self._start_section
            step = item.step
            item = slice(start, stop, step)
        else:
            raise TypeError("Item type is not <int> nor <slice>, type",
                            type(item))
        retrieved = self.group_data[item, :, self._channel_filter]
        filtered = retrieved
        return filtered

    def standarized(self):
        """Returns a standarised version of itself by wrapping it in
        StandarizedDataSection
        
        Returns:

        """
        return StandarizedDataSection(self)

    def __iter__(self):
        self.__iter_index = 0
        return self

    def __next__(self):
        if self.__iter_index < len(self):
            element = self[self.__iter_index]
            self.__iter_index += 1
            return element
        else:
            raise StopIteration()


class StandarizedDataSection(object):
    """Standarises data from a DataSection object
    
    It wraps an instance of DataSection to perform standarisation on the fly
    Basically it just computes the following operation on data x
    
        normalised = (x - x.min())/(x.max() - x.min())
        # normalised is between 0 and 1
        standarised = (normalised - 0.5) * 2
        # standarised is between -1 and +1
    """

    def __init__(self, data_section):
        """
        
        Args:
            data_section: instance of DataSection to be wrapped 
        """
        self._data_section = data_section
        self.shape = self._data_section.shape
        self.attrs = self._data_section.attrs

        # current_min = np.amin(self._data_section[0], axis=0)
        # current_max = np.amax(self._data_section[0], axis=0)
        # # find global minimum and maximum
        # import sys
        # i = 0
        # for chunk in self._data_section:
        #     sys.stdout.write("{}".format(i))
        #     sys.stdout.flush()
        #     sys.stdout.write("\r")
        #     i +=1
        #     current_max = max(current_max, np.amax(chunk, axis=0))
        #     current_min = min(current_min, np.amin(chunk, axis=0))
        # self._max = current_max
        # self._min = current_min
        self._max = data_section.group_data.attrs['train_max']
        self._min = data_section.group_data.attrs['train_min']

    def __len__(self):
        return  len(self._data_section)

    def __getitem__(self, item):
        """Gets data but normalises it before returning it
        
        Args:
            item: 

        Returns:

        """
        raw_value = self._data_section[item]
        normalised = (raw_value - self._min)/(self._max - self._min)
        standarised = (normalised - 0.5) * 2.0
        return standarised
