"""This file generate samples from a model and a test dataset

"""

import json
from models.autoencoders import DeepDoubleAutoencoderGenerator

def get_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate samples")
    # arguments for gather_dataset_online


    # define model
    parser.add_argument("load_model", metavar="load-model", help="Model to " \
                                                                "load",
                        type=str)

    # execution
    parser.add_argument("--batch-size", help="Batch size", type=int,
                        default=100)
    parser.add_argument("--step", help="The step between samples", type=int,
                        default=None)
    parser.add_argument("--num-average", help="The step between samples",
                        type=int,
                        default=1)

    # validation and test
    parser.add_argument("--out-targets", help="Records the targets",
                        type=str)
    parser.add_argument("--out-predictions", "--id", help="Records "
                                                          "predictions",
                        type=str,
                        default="")

    dargs = parser.parse_args()
    return dargs


def main():
    dargs = get_args()

    if not dargs.load_model[-1] == '/': dargs.load_model += '/'
    model_config = json.load(open(dargs.load_model + "model.json", 'r'))

    # load model
    model = DeepDoubleAutoencoderGenerator.load(dargs.load_model)

    model.recover_audio(batch_size=dargs.batch_size,
                        num_average=dargs.num_average,
                        test_out=dargs.out_targets,
                        predict_out=dargs.out_predictions)

if __name__ == "__main__":
    main()