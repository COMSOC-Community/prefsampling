from unittest import TestCase

from prefsampling.core.euclidean import EuclideanSpace, euclidean_space_to_sampler
from prefsampling.ordinal.euclidean import euclidean
from prefsampling.point import ball_uniform, ball_resampling, cube, gaussian
from tests.utils import TestSampler


def all_test_samplers_euclidean():
    def test_ball_uniform_1d(num_points, num_dimensions=1, seed=None):
        return ball_uniform(num_points, 1, seed=seed)

    def gaussian_one_point(num_dimensions, seed=None):
        return gaussian(1, num_dimensions, seed=seed)[0]

    def test_ball_resampling_2d(num_points, num_dimensions=2, seed=None):
        return ball_resampling(
            num_points,
            2,
            gaussian_one_point,
            {"num_dimensions": num_dimensions, "seed": seed},
            max_numer_resampling=20,
            seed=seed,
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
    for point_sampler, num_dim in all_point_samplers:
        samplers.append(
            TestSampler(
                euclidean,
                {
                    "num_dimensions": num_dim,
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
                {"pos_sampler": point_sampler, "num_dimensions": num_dim},
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


class TestOrdinalEuclidean(TestCase):
    def test_weak_orders(self):
        num_candidates = 5
        weak_votes = euclidean(
            10,
            num_candidates,
            2,
            EuclideanSpace.UNIFORM_BALL,
            EuclideanSpace.UNIFORM_CUBE,
            tie_radius=0.4,
        )

        assert all([sum(len(c) for c in v) == num_candidates] for v in weak_votes)

        with self.assertRaises(ValueError):
            euclidean(
                10,
                num_candidates,
                2,
                EuclideanSpace.UNIFORM_BALL,
                EuclideanSpace.UNIFORM_CUBE,
                tie_radius=0,
            )

        with self.assertRaises(ValueError):
            euclidean(
                10,
                num_candidates,
                2,
                EuclideanSpace.UNIFORM_BALL,
                EuclideanSpace.UNIFORM_CUBE,
                tie_radius=-0.4,
            )

    def test_bad_positions(self):
        with self.assertRaises(ValueError):
            euclidean(
                10,
                2,
                2,
                EuclideanSpace.UNIFORM_CUBE,
                lambda num_points, num_dimensions, seed=None: 1,
            )

        with self.assertRaises(ValueError):
            euclidean(
                1,
                2,
                2,
                EuclideanSpace.UNIFORM_CUBE,
                [[0.4, 0.2, 0.1], [0.1, 0.3, 0.8]],
            )

    def test_euclidean_space_to_sampler(self):
        with self.assertRaises(ValueError):
            euclidean_space_to_sampler("Bonjour", 2)
