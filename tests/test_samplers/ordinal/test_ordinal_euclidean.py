from prefsampling.ordinal.euclidean import euclidean
from prefsampling.point import ball_uniform, ball_resampling, cube, gaussian


def random_ord_euclidean_samplers():
    all_point_samplers = [
        lambda num_points, seed=None: ball_uniform(num_points, 1, seed=seed),
        lambda num_points, seed=None: ball_resampling(
            num_points, 2, lambda: gaussian(1, 2)[0], {}, seed=seed
        ),
        lambda num_points, seed=None: cube(num_points, 3, seed=seed),
        lambda num_points, seed=None: gaussian(num_points, 4, seed=seed),
    ]

    samplers = [
        lambda num_voters, num_candidates, seed=None: euclidean(
            num_voters,
            num_candidates,
            point_sampler=sampler,
            point_sampler_args={},
            seed=seed,
        )
        for sampler in all_point_samplers
    ]
    samplers += [
        lambda num_voters, num_candidates, seed=None: euclidean(
            num_voters,
            num_candidates,
            voters_positions=sampler(num_voters, seed),
            candidates_positions=sampler(num_candidates, seed),
            seed=seed,
        )
        for sampler in all_point_samplers
    ]
    samplers += [
        lambda num_voters, num_candidates, seed=None: euclidean(
            num_voters,
            num_candidates,
            point_sampler=sampler1,
            point_sampler_args={},
            candidate_point_sampler=sampler2,
            candidate_point_sampler_args={},
            seed=seed,
        )
        for sampler1 in all_point_samplers
        for sampler2 in all_point_samplers
    ]
    return samplers
