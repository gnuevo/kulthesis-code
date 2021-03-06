from models.simple_autoencoder_with_generator import AutoencoderWithGenerator

# print(__name__)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Autoencoder with generator")
    # arguments for gather_dataset_online
    parser.add_argument("hdf5_file", help="File containing the dataset file", type=str)
    parser.add_argument("--hdf5-group", help="Group name with the dataset", type=str)
    parser.add_argument("--dim", help="Input dimensions. Number of audio "
                                      "samples per sample", type=int,
                        required=True)
    parser.add_argument("--encoded-size", help="The encoded size", type=int,
                        default=32)
    parser.add_argument("--batch", help="Batch size", type=int, default=10000)
    parser.add_argument("--step", help="The step between samples", type=int,
                        default=None)
    dargs = parser.parse_args()

    hdf5_file = dargs.hdf5_file
    hdf5_group = dargs.hdf5_group
    dimension = dargs.dim
    encoded_size = dargs.encoded_size
    batch_size = dargs.batch
    step = dargs.step

    autoencoder = AutoencoderWithGenerator(hdf5_file, hdf5_group,
                                           input_dimension=(dimension, 2),
                                           encoding_dim=encoded_size)
    print(autoencoder)
    # autoencoder.fit_generator(batch_size=batch_size, step=step, epochs=2)
    autoencoder.train_dataset(batch_size=batch_size, step=step, epochs=2)