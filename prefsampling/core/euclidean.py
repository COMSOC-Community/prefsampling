from enum import Enum

import numpy as np

from prefsampling.inputvalidators import validate_num_voters_candidates


class EuclideanSpace(Enum):
    """
    Constants used to represent Euclidean spaces
    """

    UNIFORM = 1
    """
    Uniform space
    """
    GAUSSIAN = 2
    """
    Gaussian space
    """
    SPHERE = 3
    """
    Spheric space
    """
    BALL = 4
    """
    Ball space
    """


@validate_num_voters_candidates
def election_positions(
    num_voters: int,
    num_candidates: int,
    space: EuclideanSpace,
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
    space : :py:class:`~prefsampling.core.euclidean.EuclideanSpace`
        Type of space considered. Should be a constant defined in the
        :py:class:`~prefsampling.core.euclidean.EuclideanSpace`.
    dimension : int
        Number of dimensions for the space considered.
    rng : np.random.Generator
        The numpy generator to use for randomness.

    Returns
    -------
    (np.ndarray, np.ndarray)
        The position of the voters and of the candidates respectively.
    """
    if space == EuclideanSpace.UNIFORM:
        voters = rng.random((num_voters, dimension))
        candidates = rng.random((num_candidates, dimension))
    elif space == EuclideanSpace.GAUSSIAN:
        voters = rng.normal(loc=0.5, scale=0.15, size=(num_voters, dimension))
        candidates = rng.normal(loc=0.5, scale=0.15, size=(num_candidates, dimension))
    elif space == EuclideanSpace.SPHERE:
        voters = np.array(
            [list(random_sphere(dimension, rng)[0]) for _ in range(num_voters)]
        )
        candidates = np.array(
            [list(random_sphere(dimension, rng)[0]) for _ in range(num_candidates)]
        )
    elif space == EuclideanSpace.BALL:
        voters = np.array(
            [list(random_ball(dimension, rng)[0]) for _ in range(num_voters)]
        )
        candidates = np.array(
            [list(random_ball(dimension, rng)[0]) for _ in range(num_candidates)]
        )
    else:
        raise ValueError(
            "The `space` argument needs to be one of the constant defined in the "
            "core.euclidean.EuclideanSpace enumeration. Choices are: "
            + ", ".join(str(s) for s in EuclideanSpace)
        )
    return voters, candidates


def random_ball(
    dimension: int, rng: np.random.Generator, num_points: int = 1, radius: float = 1
) -> np.ndarray:
    random_directions = rng.normal(size=(dimension, num_points))
    random_directions /= np.linalg.norm(random_directions, axis=0)
    random_radii = rng.random(num_points) ** (1 / dimension)
    x = radius * (random_directions * random_radii).T
    return x


def random_sphere(
    dimension: int, rng: np.random.Generator, num_points: int = 1, radius: float = 1
) -> np.ndarray:
    random_directions = rng.normal(size=(dimension, num_points))
    random_directions /= np.linalg.norm(random_directions, axis=0)
    random_radii = 1.0
    return radius * (random_directions * random_radii).T
