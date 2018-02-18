"""Functions to apply to the training and testing

"""

import numpy as np


def linear_discretisation(sample, bins=np.linspace(-1.0,1.0,num=256)):
    digits = np.digitize(sample, bins)
    return (digits / float(len(bins) - 1.0)) * 2.0 - 1.0


def mu_law_encoding(sample, channels=256, mu=255.0):
    # mu = channels - 1
    mu_encoded = np.sign(sample)* np.log(1.0 +  mu * np.abs(sample)) / \
                 np.log(1 + mu)
    discrete_mu = linear_discretisation(mu_encoded, bins=np.linspace(-1.0,
                                                                     1.0,
                                                                     num=channels))
    return discrete_mu


def mu_law_decoding(sample, channels=256, mu=255.0):
    mu_decoded = np.sign(sample) / mu * (np.power(1.0 + mu, sample) - 1.0)
    return mu_decoded