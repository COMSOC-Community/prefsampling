from unittest import TestCase

from prefsampling.approval.euclidean import (
    euclidean_threshold,
    euclidean_vcr,
    euclidean_constant_size,
)
from prefsampling.point import ball_uniform, ball_resampling, cube, gaussian
from tests.utils import float_parameter_test_values, TestSampler


def all_test_samplers_approval_euclidean():
    def test_ball_uniform_1d(num_points, seed=None):
        return ball_uniform(num_points, 1, seed=seed)

    def test_ball_resampling_2d(num_points, seed=None):
        return ball_resampling(num_points, 2, lambda: gaussian(1, 2)[0], {}, seed=seed)

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
        num_voters, num_candidates, main_sampler, pos_sampler, seed=None, **kwargs
    ):
        v_pos = pos_sampler(num_voters, seed=seed)
        c_pos = pos_sampler(num_candidates, seed=seed)
        return main_sampler(
            num_voters,
            num_candidates,
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
            for point_sampler in all_point_samplers:
                params1 = {"point_sampler": point_sampler, "point_sampler_args": {}}
                params1.update(extra_params)
                samplers.append(TestSampler(euclidean_sampler, params1))

                params2 = {
                    "main_sampler": euclidean_sampler,
                    "pos_sampler": point_sampler,
                }
                params2.update(extra_params)
                samplers.append(TestSampler(euclidean_positions, params2))

                params3 = {
                    "point_sampler": point_sampler,
                    "point_sampler_args": {},
                    "candidate_point_sampler": point_sampler,
                    "candidate_point_sampler_args": {},
                }
                params3.update(extra_params)
                samplers.append(TestSampler(euclidean_sampler, params3))
    return samplers


class TestApprovalEuclidean(TestCase):
    def test_euclidean_threshold(self):
        with self.assertRaises(ValueError):
            euclidean_threshold(4, 5, 0)
        with self.assertRaises(ValueError):
            euclidean_threshold(4, 5, 0.9)

    def test_euclidean_vcr(self):
        with self.assertRaises(ValueError):
            euclidean_vcr(4, 5, voters_radius=[0, 1], candidates_radius=0)
        with self.assertRaises(ValueError):
            euclidean_vcr(4, 5, candidates_radius=[0, 1], voters_radius=0)
