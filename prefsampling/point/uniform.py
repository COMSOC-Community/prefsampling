import numpy as np

from prefsampling.inputvalidators import validate_int


def uniform(num_points: int, dimension: int, seed: int = None):
    validate_int(num_points, "num_points", 0)
    validate_int(dimension, "dimension", 1)
    rng = np.random.default_rng(seed)
    return rng.random((num_points, dimension))
