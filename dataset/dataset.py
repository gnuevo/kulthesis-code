"""

"""
import h5py
import math
import json


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