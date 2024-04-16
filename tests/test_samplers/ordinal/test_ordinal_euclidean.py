from prefsampling.core.euclidean import EuclideanSpace
from prefsampling.ordinal.euclidean import euclidean
from prefsampling.point import ball_uniform, ball_resampling, cube, gaussian
from tests.utils import TestSampler


def all_test_samplers_euclidean():
    def test_ball_uniform_1d(num_points, seed=None):
        return ball_uniform(num_points, 1, seed=seed)

    def test_ball_resampling_2d(num_points, seed=None):
        return ball_resampling(num_points, 2, lambda seed=None: gaussian(1, 2, seed=seed)[0], {}, seed=seed)

    def test_cube_3d(num_points, seed=None):
        return cube(num_points, 3, seed=seed)

    def test_gaussian_4d(num_points, seed=None):
        return gaussian(num_points, 4, seed=seed)

    all_point_samplers = [
        test_ball_uniform_1d,
        test_ball_resampling_2d,
        test_cube_3d,
        test_gaussian_4d,
    ]

    def euclidean_positions(
        num_voters, num_candidates, pos_sampler, seed=None, **kwargs
    ):
        v_pos = pos_sampler(num_voters, seed=seed)
        c_pos = pos_sampler(num_candidates, seed=seed)
        return euclidean(
            num_voters,
            num_candidates,
            voters_positions=v_pos,
            candidates_positions=c_pos,
            seed=seed,
            **kwargs,
        )

    samplers = []
    for point_sampler in all_point_samplers:
        samplers.append(
            TestSampler(
                euclidean,
                {
                    "point_sampler": point_sampler,
                    "point_sampler_args": {},
                    "candidate_point_sampler": point_sampler,
                    "candidate_point_sampler_args": {},
                },
            )
        )
        samplers.append(
            TestSampler(
                euclidean, {"point_sampler": point_sampler, "point_sampler_args": {}}
            )
        )
        samplers.append(
            TestSampler(euclidean_positions, {"pos_sampler": point_sampler})
        )
    for space in EuclideanSpace:
        samplers.append(
            TestSampler(euclidean, {"euclidean_space": space, "num_dimensions": 2})
        )
    return samplers
