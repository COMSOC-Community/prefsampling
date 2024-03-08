import numpy as np

from unittest import TestCase

from prefsampling.core import (
    resample_as_central_vote,
    rename_candidates,
    permute_voters,
    mixture,
)

from prefsampling.ordinal import (
    plackett_luce,
    group_separable,
    TreeSampler,
    urn,
    impartial_anonymous,
    impartial,
    stratification,
    single_crossing,
    single_crossing_impartial,
    single_peaked_walsh,
    single_peaked_conitzer,
    single_peaked_circle,
    mallows,
    norm_mallows,
    euclidean,
    didi,
)

from prefsampling.point import (
    uniform as point_uniform,
    ball as point_ball,
    sphere as point_sphere,
    gaussian as point_gaussian,
)

ALL_ORDINAL_SAMPLERS = [
    impartial,
    impartial_anonymous,
    lambda num_voters, num_candidates, seed=None: stratification(
        num_voters, num_candidates, 0.5, seed
    ),
    lambda num_voters, num_candidates, seed=None: urn(
        num_voters, num_candidates, 0.1, seed
    ),
    single_peaked_conitzer,
    single_peaked_circle,
    single_peaked_walsh,
    single_crossing,
    single_crossing_impartial,
    lambda num_voters, num_candidates, seed=None: mallows(
        num_voters, num_candidates, 0.5, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: norm_mallows(
        num_voters, num_candidates, 0.5, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: euclidean(
        num_voters, num_candidates, point_sampler=point_uniform, point_sampler_args={"dimension": 2}, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: euclidean(
        num_voters, num_candidates, point_sampler=point_gaussian, point_sampler_args={"dimension": 2}, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: euclidean(
        num_voters, num_candidates, point_sampler=point_sphere, point_sampler_args={"dimension": 2}, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: euclidean(
        num_voters, num_candidates, point_sampler=point_ball, point_sampler_args={"dimension": 2}, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: euclidean(
        num_voters, num_candidates, point_sampler=point_uniform, point_sampler_args={"dimension": 2}, candidate_point_sampler=point_ball, candidate_point_sampler_args={"dimension": 2}, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: euclidean(
        num_voters, num_candidates, point_sampler=point_gaussian, point_sampler_args={"dimension": 2}, candidate_point_sampler=point_ball, candidate_point_sampler_args={"dimension": 2}, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: euclidean(
        num_voters, num_candidates, point_sampler=point_sphere, point_sampler_args={"dimension": 2}, candidate_point_sampler=point_ball, candidate_point_sampler_args={"dimension": 2}, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: euclidean(
        num_voters, num_candidates, point_sampler=point_ball, point_sampler_args={"dimension": 2}, candidate_point_sampler=point_ball, candidate_point_sampler_args={"dimension": 2}, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: plackett_luce(
        num_voters, num_candidates, [1] * num_candidates, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: didi(
        num_voters, num_candidates, [1] * num_candidates, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: group_separable(
        num_voters, num_candidates, TreeSampler.SCHROEDER, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: group_separable(
        num_voters, num_candidates, TreeSampler.SCHROEDER_LESCANNE, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: group_separable(
        num_voters, num_candidates, TreeSampler.SCHROEDER_UNIFORM, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: group_separable(
        num_voters, num_candidates, TreeSampler.CATERPILLAR, seed=seed
    ),
    # lambda num_voters, num_candidates, seed=None: group_separable(
    #     num_voters, num_candidates, TreeSampler.BALANCED, seed=seed
    # ),
    lambda num_voters, num_candidates, seed=None: resample_as_central_vote(
        single_crossing(num_voters, num_candidates),
        norm_mallows,
        {"norm_phi": 0.4, "seed": seed, "num_candidates": num_candidates},
    ),
    lambda num_voters, num_candidates, seed=None: rename_candidates(
        single_crossing(num_voters, num_candidates), seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: permute_voters(
        single_crossing(num_voters, num_candidates), seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: mixture(
        num_voters,
        num_candidates,
        [norm_mallows, norm_mallows, norm_mallows],
        [4, 10, 3],
        [{"norm_phi": 0.4}, {"norm_phi": 0.9}, {"norm_phi": 0.23}],
    ),
    lambda num_voters, num_candidates, seed=None: mixture(
        num_voters,
        num_candidates,
        [
            single_crossing,
            single_peaked_circle,
            single_peaked_walsh,
        ],
        [0.5, 0.2, 0.3],
        [{}, {}, {}],
    ),
]


class TestOrdinalSamplers(TestCase):
    def helper_test_all_ordinal_samplers(self, sampler, num_voters, num_candidates):
        result = sampler(num_voters, num_candidates)

        # Test if the function returns a numpy array
        self.assertIsInstance(result, np.ndarray)

        # Test if the shape of the returned array is correct
        self.assertEqual(result.shape, (num_voters, num_candidates))

        # Test if the values are within the range of candidates
        for vote in result:
            self.assertTrue(set(vote) == set(range(num_candidates)))

        # Test if the value are int
        for vote in result:
            for candidate in vote:
                self.assertTrue(int(candidate) == candidate)

    def test_all_ordinal_samplers(self):
        num_voters = 200
        num_candidates = 5

        for sampler in ALL_ORDINAL_SAMPLERS:
            for test_sampler in [
                sampler,
                lambda x, y: sampler(num_voters=x, num_candidates=y),
                lambda x, y: sampler(x, y, seed=363),
            ]:
                with self.subTest(sampler=test_sampler):
                    self.helper_test_all_ordinal_samplers(
                        test_sampler, num_voters, num_candidates
                    )
