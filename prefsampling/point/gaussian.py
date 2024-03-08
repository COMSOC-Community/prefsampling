import numpy as np

from prefsampling.inputvalidators import validate_int


def gaussian(num_points: int, dimension: int, center_point: float = 0.5, width: float = 0.15, seed: int = None):
    validate_int(num_points, "num_points", 0)
    validate_int(dimension, "dimension", 1)
    rng = np.random.default_rng(seed)
    return rng.normal(loc=center_point, scale=width, size=(num_points, dimension))
