import numpy as np

from prefsampling.decorators import validate_num_voters_candidates


@validate_num_voters_candidates
def euclidean(
    num_voters: int = None,
    num_candidates: int = None,
    dim: int = 2,
    space: str = "uniform",
    radius: float = 0,
    seed: int = None,
) -> list[set]:
    """
    Generates approval votes from euclidean model.

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        dim : int
            Number of Dimensions.
        space : int
            Type of distribution for voters and candidates.
            {'uniform', 'gaussian', 'sphere'}
        radius : float
            The radius.
        seed : int
            Seed for numpy random number generator.

    Returns
    -------
        list[set]
            Approval votes.

     Raises
     ------
         ValueError
             When `space` not in {'uniform', 'gaussian', 'sphere'}.
    """
    if space not in {"uniform", "gaussian", "sphere"}:
        raise ValueError(f"No such type_id as {space}")

    rng = np.random.default_rng(seed)

    votes = [set() for _ in range(num_voters)]

    if space == "uniform":
        voters = rng.random((num_voters, dim))
        candidates = rng.random((num_candidates, dim))
    elif space == "gaussian":
        voters = rng.normal(loc=0.5, scale=0.15, size=(num_voters, dim))
        candidates = rng.normal(loc=0.5, scale=0.15, size=(num_candidates, dim))
    elif space == "sphere":
        voters = np.array([list(random_sphere(dim, rng)[0]) for _ in range(num_voters)])
        candidates = np.array(
            [list(random_sphere(dim, rng)[0]) for _ in range(num_candidates)]
        )

    for v in range(num_voters):
        for c in range(num_candidates):
            if radius >= np.linalg.norm(voters[v] - candidates[c]):
                votes[v].add(c)

    return votes


def random_sphere(dimension, rng, num_points=1, radius=1):
    random_directions = rng.normal(size=(dimension, num_points))
    random_directions /= np.linalg.norm(random_directions, axis=0)
    random_radii = 1.0
    return radius * (random_directions * random_radii).T
