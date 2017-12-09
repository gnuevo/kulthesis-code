"""
timbre
window size
internal size
"""

import time
import os
import itertools
from models.simple_autoencoder_with_generator import AutoencoderWithGenerator
from tensorflow.python.framework.errors_impl import ResourceExhaustedError

class Experiment1(object):
    """Performs a combination of experiments changing parameters over only one
    timbre
    
    """
    def __init__(self, configuration):
        """Initialises the object
        
        Args:
            configuration: dictionary with all the keys needed
        """
        self.__configuration = configuration
        self.__format_configuration()

    def __format_configuration(self):
        if self.__configuration["tblogdir"] is not None:
            if not self.__configuration["tblogdir"][-1] == "/":
                self.__configuration["tblogdir"] = self.__configuration[
                    "tblogdir"] + "/"
        print(self.__configuration["tblogdir"])
        if self.__configuration["model_dir"] is not None:
            if not self.__configuration["model_dir"][-1] == "/":
                self.__configuration["model_dir"] = self.__configuration[
                    "model_dir"] + "/"

    def __create_directory(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def __create_string(self, input_size, hidden_size, activation,
                        optimizer, loss):
        string = "input{}_hidden{}_{}_{}_{}".format(input_size, hidden_size,
                                                    activation, optimizer, loss)
        return string

    def run_experiment(self):
        config = self.__configuration

        current_time = time.strftime("%Y-%m-%d_%H%M%S")
        print(config["hdf5_file"])
        file_name = config["hdf5_file"].split("/")[-1].split(".")[-2]
        name = current_time + "_{}_{}".format(config["hdf5_group"], file_name)

        # create directories to store the models and tensorboard summaries
        if config["tblogdir"] is not None:
            self.__create_directory(config["tblogdir"] + name)
        if config["model_dir"] is not None:
            self.__create_directory(config["model_dir"] + name)

        elements = [self.__configuration["dimension"],
                    self.__configuration["encoded_size"],
                    self.__configuration["activation"],
                    self.__configuration["optimizer"],
                    self.__configuration["loss"]]
        print(elements)
        all_combinations = list(itertools.product(*elements))
        for c in all_combinations:
            print(c)
        for input_size, hidden_size, activation, optimizer, loss in all_combinations:
            string_name = self.__create_string(input_size, hidden_size,
                                               activation, optimizer, loss)
            print(">>>>STRING NAME", string_name)
            autoencoder = AutoencoderWithGenerator(config["hdf5_file"],
                                                   config["hdf5_group"],
                                                   input_dimension=(input_size, 2),
                                                   encoding_dim=hidden_size)

            if config["tblogdir"] is not None:
                autoencoder.callback_add_tensorboard(
                    log_dir=config["tblogdir"]+name+"/"+string_name,
                    batch_freq=config["tbbatch_freq"],
                    variables=['loss', 'val_loss'])

            if config["early_stopping_patience"] is not None:
                autoencoder.callback_add_earlystopping(patience=config[
                    "early_stopping_patience"])

            if config["model_dir"] is not None:
                autoencoder.callback_add_modelcheckpoint(config[
                                                             "model_dir"]+name+".h5py")

            everything_ok = False
            batch_size = config["batch_size"]
            while not everything_ok:
                try:
                    # run autoencoder
                    autoencoder.train_dataset(batch_size=batch_size,
                                              epochs=config["epochs"],
                                              step=config["step"],
                                              validation=config["validation"])
                    everything_ok = True
                except ResourceExhaustedError as e:
                    # force restart with lower batch_size
                    everything_ok = False
                    batch_size -= 10
                    print("Restarting with batch", batch_size)
                except Exception as e:
                    print("Exception detected", type(e), e)
                    print(e)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Help for the big experiment, it is intended to execute a"
                    "series of combinations for only one timbre")
    # arguments for gather_dataset_online
    # define dataset
    parser.add_argument("hdf5_file", help="File containing the dataset file",
                        type=str)
    parser.add_argument("--hdf5-group", help="Group name with the dataset",
                        type=str)

    # define model
    parser.add_argument("--dim", help="Input dimensions. Number of audio "
                                      "samples per sample (or list of them)",
                        type=int,
                        required=True, action='append', nargs='*')
    parser.add_argument("--encoded-size", help="The encoded size", type=int,
                        required=True, action='append', nargs='*')
    parser.add_argument("--activation", help="Desired activation, or list of them",
                        type=str, action='append', nargs='*', default=['relu'])
    parser.add_argument("--optimizer",
                        help="Desired optimizer, or list of them",
                        type=str, action='append', nargs='*',
                        default=['adadelta'])
    parser.add_argument("--loss",
                        help="Desired loss, or list of them",
                        type=str, action='append', nargs='*',
                        default=['binary_crossentropy'])

    # execution
    parser.add_argument("--batch", help="Batch size", type=int, default=10000)
    parser.add_argument("--step", help="The step between samples", type=int,
                        default=None)
    parser.add_argument("-e", "--epochs", help="The number of epochs to "
                                               "compute", type=int, default=1)

    # validation
    parser.add_argument("--validate", help="Performs validation after each "
                                           "epoch", action="store_true")

    # callbacks
    parser.add_argument("--tblogdir", help="logdir of tensorboard",
                        type=str, default=None)
    parser.add_argument("--tbbatchfreq", help="Number in batches to "
                                              "write results to disk",
                        type=int,
                        default=None)
    parser.add_argument("--model_dir", help="In this directory the models "
                                            "will be saved", type=str,
                        default=None)
    parser.add_argument("--early_stopping_patience", help="Adds early "
                                                          "stopping callback with the specified patience",
                        type=int, default=None)

    dargs = parser.parse_args()

    hdf5_file = dargs.hdf5_file
    hdf5_group = dargs.hdf5_group

    dimension = dargs.dim[0]
    encoded_size = dargs.encoded_size[0]
    print("encoded size", encoded_size)

    activation = dargs.activation[1]
    optimizer = dargs.optimizer[1]
    loss = dargs.loss[1]

    batch_size = dargs.batch
    step = dargs.step
    epochs = dargs.epochs

    log_dir = dargs.tblogdir
    batch_freq = dargs.tbbatchfreq
    validation = dargs.validate

    config = {
        "hdf5_file": hdf5_file,
        "hdf5_group": hdf5_group,
        "dimension": dimension,
        "encoded_size": encoded_size,
        "activation": activation,
        "optimizer": optimizer,
        "loss": loss,
        "batch_size": batch_size,
        "step": step,
        "epochs": epochs,
        "tblogdir": log_dir,
        "tbbatch_freq": batch_freq,
        "model_dir": dargs.model_dir,
        "early_stopping_patience": dargs.early_stopping_patience,
        "validation": validation
    }

    experiment = Experiment1(config)
    experiment.run_experiment()