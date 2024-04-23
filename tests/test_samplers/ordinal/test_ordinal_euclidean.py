from prefsampling.core.euclidean import EuclideanSpace
from prefsampling.ordinal.euclidean import euclidean
from prefsampling.point import ball_uniform, ball_resampling, cube, gaussian
from tests.utils import TestSampler


def all_test_samplers_euclidean():
    def test_ball_uniform_1d(num_points, num_dimensions=1, seed=None):
        return ball_uniform(num_points, 1, seed=seed)

    def test_ball_resampling_2d(num_points, num_dimensions=2, seed=None):
        return ball_resampling(
            num_points, 2, lambda seed=None: gaussian(1, 2, seed=seed)[0], {}, seed=seed
        )

    def test_cube_3d(num_points, num_dimensions=3, seed=None):
        return cube(num_points, 3, seed=seed)

    def test_gaussian_4d(num_points, num_dimensions=4, seed=None):
        return gaussian(num_points, 4, seed=seed)

    all_point_samplers = [
        (test_ball_uniform_1d, 1),
        (test_ball_resampling_2d, 2),
        (test_cube_3d, 3),
        (test_gaussian_4d, 4),
    ]

    def euclidean_positions(
        num_voters, num_candidates, num_dimensions, pos_sampler, seed=None, **kwargs
    ):
        v_pos = pos_sampler(num_voters, num_dimensions, seed=seed)
        c_pos = pos_sampler(num_candidates, num_dimensions, seed=seed)
        return euclidean(
            num_voters,
            num_candidates,
            num_dimensions=num_dimensions,
            voters_positions=v_pos,
            candidates_positions=c_pos,
            seed=seed,
            **kwargs,
        )

    samplers = []
    for point_sampler, num_dimensions in all_point_samplers:
        samplers.append(
            TestSampler(
                euclidean,
                {
                    "num_dimensions": num_dimensions,
                    "voters_positions": point_sampler,
                    "voters_positions_args": {},
                    "candidates_positions": point_sampler,
                    "candidates_positions_args": {},
                },
            )
        )
        samplers.append(
            TestSampler(
                euclidean_positions,
                {"pos_sampler": point_sampler, "num_dimensions": num_dimensions},
            )
        )
    for space in EuclideanSpace:
        samplers.append(
            TestSampler(
                euclidean,
                {
                    "candidates_positions": space,
                    "voters_positions": space,
                    "num_dimensions": 2,
                },
            )
        )
    return samplers
