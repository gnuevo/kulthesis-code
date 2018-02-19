"""File to extract audio from the dataset

"""
from dataset.dataset import Dataset
from models.autoencoders import DeepDoubleAutoencoderGenerator
import numpy as np
import scipy.io.wavfile as wavfile
from models.functions import linear_discretisation, mu_law_encoding, mu_law_decoding

def get_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate samples")
    # arguments for gather_dataset_online

    parser.add_argument("dataset", type=str,
                        help="Dataset path")

    parser.add_argument("--batcher", type=str,
                        help="Type of the batcher to use")
    parser.add_argument("--group", type=str,
                        required=True, action='append', nargs='*',
                        help="<group>:<out>, combination of the group and "
                             "where the out file will be stored")
    parser.add_argument("--output", type=str,
                        help="Output file")
    parser.add_argument("--section", type=str,
                        help="train|validation|test")
    parser.add_argument("--batch", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--function", type=str, default="none",
                        help="Discretisation function to be applied to the "
                             "samples (none|linear|mu), default 'none'")

    dargs = parser.parse_args()
    return dargs


def get_function(function_code):
    if function_code == "none":
        f = None
        f_args = []
    elif function_code == "linear":
        f = linear_discretisation
        f_args = [np.linspace(-1.0,1.0,num=256)]
    elif function_code == "mu":
        f = mu_law_encoding
        f_args = [256, 255.0]
    else:
        f = None
        f_args = []
    return f, f_args


def create_generator(dataset_file, timbres, section, batch_size,
                     function_code=None):
    input_group, output_group = timbres
    data = Dataset(dataset_file).group(input_group).data()
    train_section = data.attrs["train_set"]
    train_section = tuple(train_section)

    autoencoder = DeepDoubleAutoencoderGenerator(dataset_file,
                                                 input_group,
                                                 dataset_file,
                                                 output_group,
                                                 input_dimension=(1000,),
                                                 middle_layers=[],
                                                 encoding_dim=100)
    generator = autoencoder.create_generator(dataset_file,
                                           input_group,
                                           dataset_file,
                                           output_group,
                                           sample_length=1000,
                                           step=None,
                                           section=train_section,
                                           left_padding=0)

    f, f_args = get_function(function_code)
    data = generator.generate_batches(batch_size=batch_size,
                                                randomise=False,
                                                function=f,
                                                function_args=f_args)
    steps = generator.get_nbatches_in_epoch(batch_size=batch_size)
    return generator, data, steps


def extract_audio(dataset, timbres, out_file, function=None):
    input_group, output_group = timbres
    generator, data, steps = create_generator(dataset, timbres, None, 128,
                                              function_code=function)
    def generator_filter():
        for elem in data:
            yield elem[0], elem[1]

    left_data, right_data = np.array([]), np.ndarray([])
    for i in range(100):
        if i%10 == 0:
            print(i)
        new_ldata, new_rdata = next(generator_filter())
        if function == "mu":
            # decode
            new_ldata = mu_law_decoding(new_ldata, channels=256, mu=255.0)
            new_rdata = mu_law_decoding(new_rdata, channels=255, mu=255.0)
        left_data = np.append(left_data, new_ldata)
        right_data = np.append(right_data, new_rdata)

    flattened_shape = left_data.shape[0]
    left_data = np.reshape(left_data, flattened_shape)
    wavfile.write(out_file + "_{}_left.wav".format(input_group), 22050, left_data)

    flattened_shape = right_data.shape[0]
    right_data = np.reshape(right_data, flattened_shape)
    wavfile.write(out_file + "_{}_right.wav".format(output_group), 22050, right_data)


def main():
    dargs = get_args()
    dataset_file = dargs.dataset
    groups = dargs.group[0]
    print("Groups", groups)

    extract_audio(dataset_file, groups, dargs.output, function=dargs.function)


if __name__ == "__main__":
    main()