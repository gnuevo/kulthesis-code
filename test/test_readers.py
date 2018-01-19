"""tests for readers.py

"""
import unittest
import numpy as np
import unittest.mock as mock
from generator import readers
from generator.generatorutils import song_section_to_chunk_section

class Reader_test(unittest.TestCase):

    @mock.patch('generator.generatorutils.song_section_to_chunk_section')
    @mock.patch.object(np.ndarray, 'attrs', {})
    def test_read_samples(self, mock_song_section_to_chunck_section):
        """Tests that samples are retrieved correctly using a controlled 
        dataset
        
        Returns:

        """
        dataset = np.arange(1000).reshape(100, 10)
        mock_song_section_to_chunck_section.return_value = (0, len(dataset))
        print(type(song_section_to_chunk_section))
        reader = readers.Reader(dataset, 6, 10, step=0, section=None,
                                buffer_size=50)
        for sample in reader.read_samples(function=None):
            print(sample)


if __name__ == "__main__":
    unittest.main()