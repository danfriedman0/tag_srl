# Utility functions for initializing neural network parameters.
# Functions adapted from Luheng He: https://github.com/luheng/deep_srl
from __future__ import division

import numpy as np

def get_orthonormal_matrix(shape, factor=1.0, seed=None, dtype=np.float32):
    """
    Taken from
       https://github.com/luheng/deep_srl/blob/master/python/neural_srl/shared/numpy_utils.py
    """
    rng = np.random.RandomState(seed)
    if shape[0] == shape[1]:
        M = rng.randn(*shape).astype(dtype)
        Q, R = np.linalg.qr(M)
        Q = Q * np.sign(np.diag(R))
        param = Q * factor
    else:
        M1 = rng.randn(shape[0], shape[0]).astype(dtype)
        M2 = rng.randn(shape[1], shape[1]).astype(dtype)
        Q1, R1 = np.linalg.qr(M1)
        Q2, R2 = np.linalg.qr(M2)
        Q1 = Q1 * np.sign(np.diag(R1))
        Q2 = Q2 * np.sign(np.diag(R2))
        n_min = min(shape[0], shape[1])
        param = np.dot(Q1[:, :n_min], Q2[:n_min, :]) * factor
    return param


def get_block_orthonormal_matrix(input_size, output_sizes,
                                 factor=1.0, seed=None, dtype=np.float32):
    """
    Creates an orthonormal matrix for each shape (input_size, output_size)
    and concatenates them on the 2nd dimension.
    """
    mats = [get_orthonormal_matrix((input_size, output_size),
                                   factor, seed, dtype)
            for output_size in output_sizes]
    return np.concatenate(mats, axis=1)
