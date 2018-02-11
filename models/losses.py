"""Custom losses to use with the networks

"""

import numpy as np
import keras.backend as K
from abc import abstractmethod

class CustomLoss(object):

    def __str__(self):
        return self.get_code()

    def get_code(self):
        """Returns an str code that identifies the loss
        
        Returns:

        """
        pass

    @abstractmethod
    def code_to_loss(code):
        split_code = code.split(':')
        cls = split_code[0]
        args = split_code[1:]
        print(code, cls, args)
        if cls == "WeightedMSE":
            output_length = int(args[0])
            weight_param = float(args[1])
            loss = WeightedMSE(output_length, weight_param=weight_param)
        else:
            # in case there is no match we assume it's just the name of the
            # loss
            loss = code
        return loss


class WeightedMSE(CustomLoss):

    __name__ = "WeightedMSE"

    def __init__(self, output_length, weight_param=0.1):
        if weight_param < 0.0 or weight_param > 1.0:
            raise ValueError("Value for weight_param must be between 0 and 1. "
                             "Gotten {}. C'mon, think of it! Negative values "
                             "turn around the purpose of this loss. And values "
                             "higher than 1 make the error of samples in the "
                             "middle of the vector to contribute negatively! "
                             "Therefore the more error the better!".format(
                weight_param))
        self.weight_param = weight_param
        self.output_length = output_length
        lin = np.linspace(0, 2 * np.pi, output_length)
        cos = np.cos(lin)
        cos = cos * weight_param
        self.error_weights = cos + 1.0

    def __call__(self, y_true, y_pred):
        return K.mean(K.square((y_pred - y_true) * self.error_weights),
                      axis=-1)

    @abstractmethod
    def get_args(self, args):
        weight_param = float(args)
        return WeightedMSE(weight_param)

    def get_code(self):
        code = ':'.join([self.__name__,
                         str(self.output_length),
                         str(self.weight_param)])
        return code


def get_weighted_mean_squared_error(output_length, weight_param=0.1):
    """Returns the weighted mean squared loss customised for the length and the
    weight param specified.
    
    The weighted_mean_squared_error() gives more importance to the edge values
    of the output vector. This is achieved by multiplying individual values 
    by a cosine signal which maximum values are on the first and the last 
    value. By using the weight_param parameter, the relative importance of 
    the edge samples with respect to the rest can be regulated. A value of 0
    is just regular mse. Higher values 
    
    The cosine mask used to achieve this is centered in 1. So on average the 
    error is multiplied by 1.
    
    Args:
        output_lenght: length of the output vector
        weight_param: 

    Returns: a customised function that you can use as a loss function

    """
    if weight_param < 0.0 or weight_param > 1.0:
        raise ValueError("Value for weight_param must be between 0 and 1. "
                         "Gotten {}. C'mon, think of it! Negative values "
                         "turn around the purpose of this loss. And values "
                         "higher than 1 make the error of samples in the "
                         "middle of the vector to contribute negatively! "
                         "Therefore the more error the better!".format(weight_param))

    lin = np.linspace(0, 2 * np.pi, output_length)
    cos = np.cos(lin)
    cos = cos * weight_param
    error_weights = cos + 1.0

    def weighted_mean_squared_error(y_true, y_pred):
        """Calculated the so called weighted mean squared error
        
        It is just like the mse but more importance is given to the edge values
        in y_pred. 
        
        Args:
            y_true: targets
            y_pred: predictions

        Returns:

        """
        return K.mean(K.square((y_pred - y_true) * error_weights), axis=-1)
    return weighted_mean_squared_error
