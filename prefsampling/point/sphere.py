import numpy as np

from prefsampling.inputvalidators import validate_int


def sphere(num_points: int, dimension: int, center_point: float = 0.5, width: float = 1, seed: int = None):
    validate_int(num_points, "num_points", 0)
    validate_int(dimension, "dimension", 1)
    rng = np.random.default_rng(seed)

    points = []
    for _ in range(num_points):
        random_directions = rng.normal(size=(dimension, 1))
        random_directions /= np.linalg.norm(random_directions, axis=0)
        random_radii = 1.0
        point = width * (random_directions * random_radii).T
        points.append(point[0])
    return np.array(points)
