import numpy as np
from numpy import linalg


def generate_ordinal_euclidean_votes(
    model: str = "euclidean",
    num_voters: int = None,
    num_candidates: int = None,
    dim: int = 2,
    space: str = "uniform",
) -> np.ndarray:
    voters = np.zeros([num_voters, dim])
    candidates = np.zeros([num_candidates, dim])
    votes = np.zeros([num_voters, num_candidates], dtype=int)
    distances = np.zeros([num_voters, num_candidates], dtype=float)

    if model == "euclidean":
        if space == "uniform":
            voters = np.random.rand(num_voters, dim)
            candidates = np.random.rand(num_candidates, dim)
        elif space == "gaussian":
            voters = np.random.normal(loc=0.5, scale=0.15, size=(num_voters, dim))
            candidates = np.random.normal(
                loc=0.5, scale=0.15, size=(num_candidates, dim)
            )
        elif space == "sphere":
            voters = np.array([list(random_sphere(dim)[0]) for _ in range(num_voters)])
            candidates = np.array(
                [list(random_sphere(dim)[0]) for _ in range(num_candidates)]
            )

    for v in range(num_voters):
        for c in range(num_candidates):
            votes[v][c] = c
            distances[v][c] = np.linalg.norm(voters[v] - candidates[c], ord=dim)

        votes[v] = [x for _, x in sorted(zip(distances[v], votes[v]))]

    return votes


# AUXILIARY
def random_ball(dimension, num_points=1, radius=1):
    random_directions = np.random.normal(size=(dimension, num_points))
    random_directions /= linalg.norm(random_directions, axis=0)
    random_radii = np.random.random(num_points) ** (1 / dimension)
    x = radius * (random_directions * random_radii).T
    return x


def random_sphere(dimension, num_points=1, radius=1):
    random_directions = np.random.normal(size=(dimension, num_points))
    random_directions /= linalg.norm(random_directions, axis=0)
    random_radii = 1.0
    return radius * (random_directions * random_radii).T
