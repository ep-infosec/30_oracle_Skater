"""Holds all the kernels"""
import numpy as np


def rbf_kernel(distance, kernel_width=1.0):
    """Passes a distance through a gaussian kernel"""
    return np.sqrt(np.exp(-(distance ** 2) / kernel_width ** 2))


def flatten(array):
    return [item for sublist in array for item in sublist]
