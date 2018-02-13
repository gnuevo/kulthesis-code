"""Functions to apply to the training and testing

"""

import numpy as np

def linear_discretisation(sample, bins=np.linspace(-1.0,1.0,num=256)):
    digits = np.digitize(sample, bins)
    return (digits / 128.0) - 0.5