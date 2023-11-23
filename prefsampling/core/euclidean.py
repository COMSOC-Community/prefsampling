import numpy as np

from prefsampling.decorators import validate_num_voters_candidates

EUCLIDEAN_SPACE_UNIFORM = 1
"""Constant representing uniform spaces for Euclidean models"""

EUCLIDEAN_SPACE_GAUSSIAN = 2
"""Constant representing Gaussian spaces for Euclidean models"""

EUCLIDEAN_SPACE_SPHERE = 3
"""Constant representing spherical spaces for Euclidean models"""


@validate_num_voters_candidates
def election_positions(
    num_voters: int,
    num_candidates: int,
    space: int,
    dimension: int,
    rng: np.random.Generator,
) -> (np.ndarray, np.ndarray):
    """
    Returns the position of the voters and the candidates in a Euclidean space.

    Parameters
    ----------
    num_voters: int
        The number of voters.
    num_candidates: int
        The number of candidates.
    space : int
        Type of space considered. Should be a constant such as
        :py:const:`~prefsampling.core.euclidean.EUCLIDEAN_SPACE_UNIFORM`
    dimension : int
        Number of dimensions for the space considered.
    rng : np.random.Generator
        The numpy generator to use for randomness.

    Returns
    -------
    (np.ndarray, np.ndarray)
        The position of the voters and of the candidates respectively.
    """
    if space == EUCLIDEAN_SPACE_UNIFORM:
        voters = rng.random((num_voters, dimension))
        candidates = rng.random((num_candidates, dimension))
    elif space == EUCLIDEAN_SPACE_GAUSSIAN:
        voters = rng.normal(loc=0.5, scale=0.15, size=(num_voters, dimension))
        candidates = rng.normal(loc=0.5, scale=0.15, size=(num_candidates, dimension))
    elif space == EUCLIDEAN_SPACE_SPHERE:
        voters = np.array(
            [list(random_sphere(dimension, rng)[0]) for _ in range(num_voters)]
        )
        candidates = np.array(
            [list(random_sphere(dimension, rng)[0]) for _ in range(num_candidates)]
        )
    else:
        raise ValueError(
            "The `space` argument needs to be one of the constant defined in the "
            "core.euclidean model (e.g., EUCLIDEAN_SPACE_UNIFORM or "
            "EUCLIDEAN_SPACE_GAUSSIAN)."
        )
    return voters, candidates


def random_ball(dimension: int, num_points: int = 1, radius: float = 1) -> float:
    random_directions = np.random.normal(size=(dimension, num_points))
    random_directions /= np.linalg.norm(random_directions, axis=0)
    random_radii = np.random.random(num_points) ** (1 / dimension)
    x = radius * (random_directions * random_radii).T
    return x


def random_sphere(
    dimension: int, rng: np.random.Generator, num_points: int = 1, radius: float = 1
) -> float:
    random_directions = rng.normal(size=(dimension, num_points))
    random_directions /= np.linalg.norm(random_directions, axis=0)
    random_radii = 1.0
    return radius * (random_directions * random_radii).T
