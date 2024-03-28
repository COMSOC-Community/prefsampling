from prefsampling.approval.euclidean import (
    euclidean_threshold,
    euclidean_vcr,
    euclidean_constant_size,
)
from prefsampling.point import ball_uniform, ball_resampling, cube, gaussian
from tests.utils import float_parameter_test_values


def random_app_euclidean_samplers():
    all_point_samplers = [
        lambda num_points, seed=None: ball_uniform(num_points, 1, seed=seed),
        lambda num_points, seed=None: ball_resampling(
            num_points, 2, lambda: gaussian(1, 2)[0], {}, seed=seed
        ),
        lambda num_points, seed=None: cube(num_points, 3, seed=seed),
        lambda num_points, seed=None: gaussian(num_points, 4, seed=seed),
    ]

    def euclidean_vcr_params():
        res = []
        for v in float_parameter_test_values(0, 10, 3):
            for c in float_parameter_test_values(0, 10, 3):
                res.append({"voters_radius": v, "candidates_radius": c})
        return res

    def euclidean_threshold_params():
        res = []
        for t in float_parameter_test_values(1, 10, 3):
            res.append({"threshold": t})
        return res

    def euclidean_constant_size_params():
        res = []
        for t in float_parameter_test_values(0, 1, 3):
            res.append({"rel_num_approvals": t})
        return res

    samplers = []
    for euclidean_sampler, params_generator in [
        (euclidean_vcr, euclidean_vcr_params),
        (euclidean_threshold, euclidean_threshold_params),
        (euclidean_constant_size, euclidean_constant_size_params),
    ]:
        for extra_params in params_generator():
            samplers += [
                lambda num_voters, num_candidates, seed=None: euclidean_sampler(
                    num_voters,
                    num_candidates,
                    point_sampler=sampler,
                    point_sampler_args={},
                    seed=seed,
                    **extra_params,
                )
                for sampler in all_point_samplers
            ]
            samplers += [
                lambda num_voters, num_candidates, seed=None: euclidean_sampler(
                    num_voters,
                    num_candidates,
                    voters_positions=sampler(num_voters, seed),
                    candidates_positions=sampler(num_candidates, seed),
                    seed=seed,
                    **extra_params,
                )
                for sampler in all_point_samplers
            ]
            samplers += [
                lambda num_voters, num_candidates, seed=None: euclidean_sampler(
                    num_voters,
                    num_candidates,
                    point_sampler=sampler1,
                    point_sampler_args={},
                    candidate_point_sampler=sampler2,
                    candidate_point_sampler_args={},
                    seed=seed,
                    **extra_params,
                )
                for sampler1 in all_point_samplers
                for sampler2 in all_point_samplers
            ]
    return samplers
