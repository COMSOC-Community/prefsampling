from unittest import TestCase

from prefsampling.approval.euclidean import (
    euclidean_threshold,
    euclidean_vcr,
    euclidean_constant_size,
)
from prefsampling.core.euclidean import EuclideanSpace
from prefsampling.point import ball_uniform, ball_resampling, cube, gaussian
from tests.utils import float_parameter_test_values, TestSampler


def all_test_samplers_approval_euclidean():
    def test_ball_uniform_1d(num_points, num_dimensions=1, seed=None):
        return ball_uniform(num_points, 1, seed=seed)

    def gaussian_one_point(num_dimensions, seed=None):
        return gaussian(1, num_dimensions, seed=seed)[0]

    def test_ball_resampling_2d(num_points, num_dimensions=2, seed=None):
        return ball_resampling(
            num_points, 2, gaussian_one_point, {'num_dimensions': num_dimensions, 'seed': seed}, max_numer_resampling=20, seed=seed
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

    def euclidean_vcr_params():
        res = []
        for v in float_parameter_test_values(0, 10, 2):
            for c in float_parameter_test_values(0, 10, 2):
                res.append({"voters_radius": v, "candidates_radius": c})
        return res

    def euclidean_threshold_params():
        res = []
        for t in float_parameter_test_values(1, 10, 2):
            res.append({"threshold": t})
        return res

    def euclidean_constant_size_params():
        res = []
        for t in float_parameter_test_values(0, 1, 2):
            res.append({"rel_num_approvals": t})
        return res

    def euclidean_positions(
        num_voters,
        num_candidates,
        num_dimensions,
        main_sampler,
        pos_sampler,
        seed=None,
        **kwargs,
    ):
        v_pos = pos_sampler(num_voters, seed=seed)
        c_pos = pos_sampler(num_candidates, seed=seed)
        return main_sampler(
            num_voters,
            num_candidates,
            num_dimensions=num_dimensions,
            voters_positions=v_pos,
            candidates_positions=c_pos,
            seed=seed,
            **kwargs,
        )

    samplers = []
    for euclidean_sampler, params_generator in [
        (euclidean_vcr, euclidean_vcr_params),
        (euclidean_threshold, euclidean_threshold_params),
        (euclidean_constant_size, euclidean_constant_size_params),
    ]:
        for extra_params in params_generator():
            for space in EuclideanSpace:
                params = {
                    "voters_positions": space,
                    "candidates_positions": space,
                    "num_dimensions": 2,
                }
                params.update(extra_params)
                samplers.append(TestSampler(euclidean_sampler, params))
            for point_sampler, num_dim in all_point_samplers:
                params1 = {
                    "num_dimensions": num_dim,
                    "voters_positions": point_sampler,
                    "voters_positions_args": {},
                    "candidates_positions": point_sampler,
                    "candidates_positions_args": {},
                }
                params1.update(extra_params)
                samplers.append(TestSampler(euclidean_sampler, params1))

                params2 = {
                    "num_dimensions": num_dim,
                    "main_sampler": euclidean_sampler,
                    "pos_sampler": point_sampler,
                }
                params2.update(extra_params)
                samplers.append(TestSampler(euclidean_positions, params2))
    return samplers


class TestApprovalEuclidean(TestCase):
    def test_euclidean_threshold(self):
        euclidean_threshold(4, 5, 2, 2, ball_uniform, ball_uniform)
        with self.assertRaises(ValueError):
            euclidean_threshold(4, 5, 0, 2, ball_uniform, ball_uniform)
        with self.assertRaises(ValueError):
            euclidean_threshold(4, 5, 0.9, 2, ball_uniform, ball_uniform)

    def test_euclidean_vcr(self):
        euclidean_vcr(4, 5, 4, 0, 2, ball_uniform, ball_uniform)
        with self.assertRaises(ValueError):
            euclidean_vcr(4, 5, [0, 1], 0, 2, ball_uniform, ball_uniform)
        with self.assertRaises(ValueError):
            euclidean_vcr(4, 5, 0, [0, 1], 2, ball_uniform, ball_uniform)

    def test_euclidean_constant_size(self):
        euclidean_constant_size(4, 5, 0.7, 2, ball_uniform, ball_uniform)
