"""This script can be run doing something like this
python3 -m run_scripts.test_awg 
    ~/MAI/MT/MasterThesis/midiutil/dataset/v5_japan_dataset.hdf5 
    --hdf5-group timbre0 --dim 20000 --encoded-size 200 
    --batch 100 --step 1000
"""
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
    parser.add_argument("-e", "--epochs", help="The number of epochs to "
                                               "compute", type=int, default=1)
    parser.add_argument("--validate", help="Performs validation after each "
                                           "epoch", action="store_true")
    parser.add_argument("--tblogdir", help="logdir of tensorboard",
                        type=str, default="/tmp/tensorboard")
    parser.add_argument("--tbbatchfreq", help="Number in batches to "
                                                    "write results to disk",
                        type=int,
                        default=None)
    dargs = parser.parse_args()

    hdf5_file = dargs.hdf5_file
    hdf5_group = dargs.hdf5_group
    dimension = dargs.dim
    encoded_size = dargs.encoded_size
    batch_size = dargs.batch
    step = dargs.step
    epochs = dargs.epochs
    log_dir = dargs.tblogdir
    batch_freq = dargs.tbbatchfreq
    validation = dargs.validate

    autoencoder = AutoencoderWithGenerator(hdf5_file, hdf5_group,
                                           input_dimension=(dimension, 2),
                                           encoding_dim=encoded_size)
    print(autoencoder)
    # autoencoder.fit_generator(batch_size=batch_size, step=step, epochs=2)
    autoencoder.callback_add_tensorboard(log_dir=log_dir,
                                         batch_freq=batch_freq, variables=[
                                            'loss', 'val_loss'])
    autoencoder.callback_add_tensorboard(log_dir=log_dir)
    autoencoder.train_dataset(batch_size=batch_size, step=step,
                              epochs=epochs, validation=validation)