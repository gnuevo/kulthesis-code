""" Experiment 4

This is just a code-cleaned version of Experiment3

Experiment 3 allows to train a network to convert from the timbre of one 
instrument to another one. Using the DeepAutoencoder
"""
from keras import losses
import time
import os
import itertools
from models.autoencoders import DeepDoubleAutoencoderGenerator
from tensorflow.python.framework.errors_impl import ResourceExhaustedError
import numpy as np
from models.functions import linear_discretisation, mu_law_encoding


def get_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Help for the big experiment3. Trains the network to "
                    "change the timbre from input instrument to output "
                    "instrument")

    # arguments for gather_dataset_online
    # define dataset
    parser.add_argument("input_group", help="Input file and group with the "
                                            "following format file:group",
                        type=str)

    parser.add_argument("output_group", help="Output file and group with the "
                                             "following format file:group",
                        type=str)

    # define model
    ## architecture
    parser.add_argument("--dim", help="Input dimensions. Number of audio "
                                      "samples per sample (or list of them)",
                        type=int,
                        required=True, action='append', nargs='*')
    parser.add_argument("--encoded-size", help="The encoded size", type=int,
                        required=True, action='append', nargs='*')
    parser.add_argument("--activation",
                        help="Desired activation, or list of them",
                        type=str, action='append', nargs='*', default=['relu'])
    parser.add_argument("--function", type=str, default=["none"],
                        action='append', nargs='*',
                        help="function to apply")

    ## learning params
    parser.add_argument("--optimizer",
                        help="Desired optimizer, or list of them",
                        type=str, action='append', nargs='*',
                        default=['adadelta'])
    parser.add_argument("--decay",
                        help="Desired decay value",
                        type=float, default=0.0)
    parser.add_argument("-r", "--learning-rate",
                        help="WARNING, THIS COULD TOTALLY F*CK UP THE "
                             "TRAINING IF YOU DON'T KNOW WHAT YOU'RE DOING"
                             "Starting "
                             "learning "
                             "rate",
                        type=float, default=None)
    parser.add_argument("--loss",
                        help="Desired loss, or list of them",
                        type=str, action='append', nargs='*',
                        default=['binary_crossentropy'])
    parser.add_argument("--error-weights", help="Triggers the use of "
                                                "error weights. Float "
                                                "between 0 and 1",
                        type=float, default=None)

    # execution
    parser.add_argument("--batch", help="Batch size", type=int, default=10000)
    parser.add_argument("--step", help="The step between samples", type=int,
                        default=None)
    parser.add_argument("-e", "--epochs", help="The number of epochs to "
                                               "compute", type=int, default=1)

    # validation and test
    parser.add_argument("--validate", help="Performs validation after each "
                                           "epoch", action="store_true")
    parser.add_argument("--test", help="Performs test after training is "
                                       "finished", action="store_true")

    # callbacks
    ## tensorboard
    parser.add_argument("--tblogdir", help="logdir of tensorboard",
                        type=str, default=None)
    parser.add_argument("--tbbatchfreq", help="Number in batches to "
                                              "write results to disk",
                        type=int,
                        default=None)
    ## save models
    parser.add_argument("--save", help="In this directory the models "
                                       "will be saved ("
                                       "customModelCheckpoint)", type=str,
                        default=None)
    parser.add_argument("--save-period", help="Period to save model, "
                                              "in epochs", type=int,
                        default=5)
    parser.add_argument("--load-model", help="Load network and continue "
                                             "training", type=str,
                        default=None)
    parser.add_argument("--early_stopping_patience", help="Adds early "
                                                          "stopping callback with the specified patience",
                        type=int, default=None)

    # others
    parser.add_argument("--identifier", "--id", help="Name of the execution, "
                                                     "helps to "
                                                     "organise them", type=str,
                        default="")
    parser.add_argument("--stereo", help="Stereo audio instead of mono",
                        action="store_true")

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
        raise RuntimeWarning("I'm almos sure you don't want to use mu-law")
        f = mu_law_encoding
        f_args = [256, 255.0]
    else:
        f = None
        f_args = []
    return f, f_args


class Experiment4(object):
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
        if self.__configuration.tblogdir is not None:
            if not self.__configuration.tblogdir[-1] == "/":
                self.__configuration.tblogdir = \
                    self.__configuration.tblogdir + "/"
        print(self.__configuration.tblogdir)

    def __create_directory(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def __create_string(self, input_size, hidden_size, activation,
                        optimizer, loss, func):
        string = "input{}_hidden{}_{}_{}_{}_{}".format(input_size, hidden_size,
                                                    activation, optimizer,
                                                    loss, func)
        return string

    def run_experiment(self):
        config = self.__configuration

        current_time = time.strftime("%Y-%m-%d_%H%M%S")
        print(config.input_file)
        file_name = config.input_file.split("/")[-1].split(".")[-2]
        name = current_time + "_{input}_{output}_{file}_{id}"\
            .format(input=config.input_group,
                    output=config.output_group,
                    file=file_name,
                    id=config.identifier)
        print(name)

        # create directories to store the models and tensorboard summaries
        if config.tblogdir is not None:
            self.__create_directory(config.tblogdir + name)

        # extract variables
        input_file = config.input_file
        input_group = config.input_group
        output_file = config.output_file
        output_group = config.output_group

        elements = [
                    config.dimension,
                    config.encoded_size,
                    config.activation,
                    config.optimizer,
                    config.loss,
                    config.function
                   ]
        print(elements)

        all_combinations = list(itertools.product(*elements))
        for c in all_combinations:
            print(c)

        for input_size, hidden_size, activation, optimizer, loss, function \
                in all_combinations:
            string_name = self.__create_string(input_size, hidden_size,
                                               activation, optimizer, loss,
                                               function)
            if config.stereo:
                input_shape = (input_size, 2)
            else:
                input_shape = (input_size,)
            print(">>>>STRING NAME", string_name)

            if config.error_weights is not None:
                from models import losses
                print("Using custom loss with param =", config.error_weights)
                if config.stereo:
                    raise ValueError("Error_weights DON'T work for 'stereo'")
                loss = losses.WeightedMSE(input_shape[0], config.error_weights)

            # create or load the model
            if config.load_model is None:
                autoencoder = DeepDoubleAutoencoderGenerator(
                                                         input_file,
                                                         input_group,
                                                         output_file,
                                                         output_group,
                                                         input_dimension=input_shape,
                                                         middle_layers=[],
                                                         encoding_dim=hidden_size)
                autoencoder.initialise(activation, optimizer, loss)
            else:
                autoencoder = DeepDoubleAutoencoderGenerator.load(
                    config.load_model)

            # program learning rate and decay
            if config.learning_rate is not None:
                raise RuntimeWarning("Learning rate defined. This can f*ck "
                                     "up the training depending on the "
                                     "optimizer. I hope you know what you're doing")
                autoencoder.callback_learning_rate_scheduler(
                                        config.learning_rate,
                                        config.decay)

            if config.tblogdir is not None:
                autoencoder.callback_add_tensorboard(
                    log_dir=config.tblogdir + name + "/" + string_name,
                    batch_freq=config.tbbatch_freq,
                    variables=['loss', 'val_loss'])

            if config.early_stopping_patience is not None:
                raise NotImplementedError("Please don't use early stopping "
                                          "option")
                autoencoder.callback_add_earlystopping(
                    patience=config.early_stopping_patience)

            if config.save is not None:
                self.__create_directory(config.save + name)
                autoencoder.callback_add_custommodelcheckpoint(
                                config.save + name + "/" + string_name,
                    period=config.save_period)

            everything_ok = False
            batch_size = config.batch_size

            f, f_args = get_function(function)
            while not everything_ok:
                try:
                    # run autoencoder
                    autoencoder.train_dataset(batch_size=batch_size,
                                              epochs=config.epochs,
                                              step=config.step,
                                              validation=config.validate,
                                              function=f,
                                              function_args=f_args)
                    everything_ok = True
                except ResourceExhaustedError as e:
                    # force restart with lower batch_size
                    everything_ok = False
                    batch_size -= 10
                    raise RuntimeWarning("The batch is too big! Error!")
                    exit(-1)
                    print("Restarting with batch", batch_size)
                # except Exception as e:
                #     print("Exception detected", type(e), e)
                #     print(e)
            # do test
            # FIXME, you should not pay attention to this part unless it's
            # the model you have selected, otherwise you're overfitting
            if config.test:
                test_error = autoencoder.test_dataset(batch_size=batch_size,
                                               step=step)
                print("Network:", input_size, hidden_size, activation, optimizer, loss)
                print("\tFor step =", step, "batch_size =", batch_size)
                print("\tHas test error", test_error)

                autoencoder.recover_audio(batch_size=batch_size)


def parse_file_group(string):
    file = string.split(':')[0]
    group = string.split(':')[1]
    return file, group


def get_argument_multiple(argument):
    """Working with multiple arguments
    
    When an argument accepts several values it gets crazy under argparse
    This code checks if the argument is has several values. If it does, it
    returns an iterable with them; if not, it returns the only element.
    
    Args:
        argument: 

    Returns:

    """
    if len(argument) == 2:
        print("len 2")
        arg = argument[1]
    elif len(argument) == 1:
        print("len 1")
        arg = argument
    else:
        print("ERROR in the argument", argument)
    return arg

def main():
    dargs = get_args()

    input_file, input_group = parse_file_group(dargs.input_group)
    output_file, output_group = parse_file_group(dargs.output_group)

    dargs.input_file = input_file
    dargs.input_group = input_group
    dargs.output_file = output_file
    dargs.output_group = output_group

    dimension = dargs.dim[0]
    dargs.dimension = dimension
    encoded_size = dargs.encoded_size[0]
    dargs.encoded_size = encoded_size
    print("encoded size", encoded_size)

    dargs.activation = get_argument_multiple(dargs.activation)
    dargs.optimizer = get_argument_multiple(dargs.optimizer)
    loss = get_argument_multiple(dargs.loss)
    dargs.loss = loss
    dargs.function = get_argument_multiple(dargs.function)

    print(dargs.activation, dargs.optimizer, loss)

    batch_size = dargs.batch
    dargs.batch_size = batch_size

    stereo = dargs.stereo


    batch_freq = dargs.tbbatchfreq
    dargs.tbbatch_freq = batch_freq


    # warnings and errors
    if stereo:
        print("ATTENTION! don't use --stereo, it's note implemented yet")
        exit(-1)

    if dargs.early_stopping_patience:
        raise ValueError("Please, don't use early_stopping_patience, I think it makes no sense")


    experiment = Experiment4(dargs)
    experiment.run_experiment()


if __name__ == "__main__":
    main()