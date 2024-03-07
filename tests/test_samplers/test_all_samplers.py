from unittest import TestCase

from prefsampling import CONSTANTS
from prefsampling.core.euclidean import EuclideanSpace
from prefsampling.core.composition import mixture
from prefsampling.core.filters import (
    resample_as_central_vote,
    permute_voters,
    rename_candidates,
)
from prefsampling.ordinal import (
    urn as ordinal_urn,
    impartial as ordinal_impartial,
    impartial_anonymous as ordinal_impartial_anonymous,
    stratification as ordinal_stratification,
    single_crossing as ordinal_single_crossing,
    single_crossing_impartial as ordinal_single_crossing_impartial,
    single_peaked_conitzer as ordinal_single_peaked_conitzer,
    single_peaked_circle as ordinal_single_peaked_circle,
    single_peaked_walsh as ordinal_single_peaked_walsh,
    euclidean as ordinal_euclidean,
    mallows as ordinal_mallows,
    norm_mallows as ordinal_norm_mallows,
    plackett_luce as ordinal_plackett_luce,
    didi as ordinal_didi,
    identity as ordinal_identity,
    group_separable as ordinal_group_separable,
    TreeSampler,
)

from prefsampling.approval import (
    resampling as approval_resampling,
    disjoint_resampling as approval_disjoint_resampling,
    moving_resampling as approval_moving_resampling,
    impartial as approval_impartial,
    impartial_constant_size as approval_impartial_constant_size,
    euclidean as approval_euclidean,
    noise as approval_noise,
    identity as approval_identity,
    full as approval_full,
    empty as approval_empty,
    urn_partylist as approval_urn_partylist,
    urn as approval_urn,
    urn_constant_size as approval_urn_constant_size,
    truncated_ordinal as approval_truncated_ordinal,
    NoiseType,
)

ALL_SAMPLERS = [
    ordinal_impartial,
    ordinal_impartial_anonymous,
    lambda num_voters, num_candidates, seed=None: ordinal_stratification(
        num_voters, num_candidates, 0.5, seed
    ),
    lambda num_voters, num_candidates, seed=None: ordinal_urn(
        num_voters, num_candidates, 0.1, seed
    ),
    ordinal_single_peaked_conitzer,
    ordinal_single_peaked_circle,
    ordinal_single_peaked_walsh,
    ordinal_single_crossing,
    ordinal_single_crossing_impartial,
    lambda num_voters, num_candidates, seed=None: ordinal_euclidean(
        num_voters, num_candidates, space=EuclideanSpace.UNIFORM, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: ordinal_euclidean(
        num_voters, num_candidates, space=EuclideanSpace.GAUSSIAN, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: ordinal_euclidean(
        num_voters, num_candidates, space=EuclideanSpace.SPHERE, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: ordinal_euclidean(
        num_voters, num_candidates, space=EuclideanSpace.BALL, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: ordinal_euclidean(
        num_voters, num_candidates, space=CONSTANTS.UNIFORM, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: ordinal_euclidean(
        num_voters, num_candidates, space=CONSTANTS.GAUSSIAN, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: ordinal_euclidean(
        num_voters, num_candidates, space=CONSTANTS.SPHERE, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: ordinal_euclidean(
        num_voters, num_candidates, space=CONSTANTS.BALL, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: ordinal_mallows(
        num_voters, num_candidates, 0.5, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: ordinal_norm_mallows(
        num_voters, num_candidates, 0.5, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: ordinal_plackett_luce(
        num_voters, num_candidates, [1] * num_candidates, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: ordinal_didi(
        num_voters, num_candidates, [1] * num_candidates, seed=seed
    ),
    ordinal_identity,
    lambda num_voters, num_candidates, seed=None: ordinal_group_separable(
        num_voters, num_candidates, TreeSampler.SCHROEDER, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: ordinal_group_separable(
        num_voters, num_candidates, TreeSampler.SCHROEDER_LESCANNE, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: ordinal_group_separable(
        num_voters, num_candidates, TreeSampler.SCHROEDER_UNIFORM, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: ordinal_group_separable(
        num_voters, num_candidates, TreeSampler.CATERPILLAR, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: ordinal_group_separable(
        num_voters, num_candidates, CONSTANTS.SCHROEDER, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: ordinal_group_separable(
        num_voters, num_candidates, CONSTANTS.SCHROEDER_LESCANNE, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: ordinal_group_separable(
        num_voters, num_candidates, CONSTANTS.SCHROEDER_UNIFORM, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: ordinal_group_separable(
        num_voters, num_candidates, CONSTANTS.CATERPILLAR, seed=seed
    ),
    # lambda num_voters, num_candidates, seed=None: ordinal_group_separable(
    #     num_voters, num_candidates, TreeSampler.BALANCED, seed=seed
    # ),
    lambda num_voters, num_candidates, seed=None: approval_resampling(
        num_voters, num_candidates, 0.5, 0.5, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: approval_disjoint_resampling(
        num_voters, num_candidates, 0.5, 0.5, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: approval_moving_resampling(
        num_voters, num_candidates, 0.5, 0.5, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: approval_impartial(
        num_voters, num_candidates, 0.5, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: approval_impartial_constant_size(
        num_voters, num_candidates, 0.5, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: approval_urn(
        num_voters, num_candidates, 0.5, 0.8, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: approval_urn_constant_size(
        num_voters, num_candidates, 0.5, 0.8, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: approval_euclidean(
        num_voters, num_candidates, space=EuclideanSpace.UNIFORM, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: approval_euclidean(
        num_voters, num_candidates, space=EuclideanSpace.GAUSSIAN, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: approval_euclidean(
        num_voters, num_candidates, space=EuclideanSpace.SPHERE, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: approval_euclidean(
        num_voters, num_candidates, space=EuclideanSpace.BALL, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: approval_euclidean(
        num_voters, num_candidates, space=CONSTANTS.UNIFORM, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: approval_euclidean(
        num_voters, num_candidates, space=CONSTANTS.GAUSSIAN, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: approval_euclidean(
        num_voters, num_candidates, space=CONSTANTS.SPHERE, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: approval_euclidean(
        num_voters, num_candidates, space=CONSTANTS.BALL, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: approval_noise(
        num_voters, num_candidates, 0.5, 0.5, noise_type=NoiseType.HAMMING, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: approval_noise(
        num_voters, num_candidates, 0.5, 0.5, noise_type=NoiseType.ZELINKA, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: approval_noise(
        num_voters, num_candidates, 0.5, 0.5, noise_type=NoiseType.JACCARD, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: approval_noise(
        num_voters,
        num_candidates,
        0.5,
        0.5,
        noise_type=NoiseType.BUNKE_SHEARER,
        seed=seed,
    ),
    lambda num_voters, num_candidates, seed=None: approval_noise(
        num_voters, num_candidates, 0.5, 0.5, noise_type=CONSTANTS.HAMMING, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: approval_noise(
        num_voters, num_candidates, 0.5, 0.5, noise_type=CONSTANTS.ZELINKA, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: approval_noise(
        num_voters, num_candidates, 0.5, 0.5, noise_type=CONSTANTS.JACCARD, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: approval_noise(
        num_voters,
        num_candidates,
        0.5,
        0.5,
        noise_type=CONSTANTS.BUNKE_SHEARER,
        seed=seed,
    ),
    lambda num_voters, num_candidates, seed=None: approval_identity(
        num_voters, num_candidates, 0.5, seed=seed
    ),
    approval_full,
    approval_empty,
    lambda num_voters, num_candidates, seed=None: approval_urn_partylist(
        num_voters, num_candidates, 0.1, 3, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: approval_truncated_ordinal(
        num_voters, num_candidates, 0.5, ordinal_urn, {"alpha": 0.8}
    ),
    lambda num_voters, num_candidates, seed=None: mixture(
        num_voters,
        num_candidates,
        [
            ordinal_single_crossing,
            ordinal_single_peaked_circle,
            ordinal_single_peaked_walsh,
        ],
        [0.5, 0.2, 0.3],
        [{}, {}, {}],
    ),
    lambda num_voters, num_candidates, seed=None: mixture(
        num_voters,
        num_candidates,
        [approval_identity, approval_full, approval_urn_partylist],
        [0.5, 0.2, 0.3],
        [{"rel_num_approvals": 0.4}, {}, {"alpha": 0.1, "parties": 3}],
    ),
    lambda num_voters, num_candidates, seed=None: mixture(
        num_voters,
        num_candidates,
        [ordinal_norm_mallows, ordinal_norm_mallows, ordinal_norm_mallows],
        [4, 10, 3],
        [{"norm_phi": 0.4}, {"norm_phi": 0.9}, {"norm_phi": 0.23}],
    ),
    lambda num_voters, num_candidates, seed=None: permute_voters(
        ordinal_single_crossing(num_voters, num_candidates), seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: permute_voters(
        approval_identity(num_voters, num_candidates, rel_num_approvals=0.5), seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: rename_candidates(
        ordinal_single_crossing(num_voters, num_candidates), seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: rename_candidates(
        approval_identity(num_voters, num_candidates, rel_num_approvals=0.5),
        seed=seed,
        num_candidates=num_candidates,
    ),
    lambda num_voters, num_candidates, seed=None: resample_as_central_vote(
        ordinal_single_crossing(num_voters, num_candidates),
        ordinal_norm_mallows,
        {"norm_phi": 0.4, "seed": seed, "num_candidates": num_candidates},
    ),
    lambda num_voters, num_candidates, seed=None: resample_as_central_vote(
        approval_identity(num_voters, num_candidates, rel_num_approvals=0.5),
        approval_resampling,
        {"phi": 0.4, "p": 0.523, "seed": seed, "num_candidates": num_candidates},
    ),
]


class TestSamplers(TestCase):
    def helper_test_all_samplers(self, sampler, num_voters, num_candidates):
        # All the necessary arguments are there
        sampler(num_voters, num_candidates)
        sampler(num_voters, num_candidates, seed=23)
        sampler(num_voters=num_voters, num_candidates=num_candidates, seed=23)

        # The samplers are decorated to exclude bad number of voters and/or candidates arguments
        with self.assertRaises(ValueError):
            sampler(1, -2)
        with self.assertRaises(ValueError):
            sampler(-2, 1)
        with self.assertRaises(ValueError):
            sampler(-2, -2)
        with self.assertRaises(TypeError):
            sampler(1.5, 2)
        with self.assertRaises(TypeError):
            sampler(1, 2.5)
        with self.assertRaises(TypeError):
            sampler(1.5, 2.5)

    def test_all_samplers(self):
        num_voters = 200
        num_candidates = 5

        for sampler in ALL_SAMPLERS:
            with self.subTest(sampler=sampler):
                self.helper_test_all_samplers(sampler, num_voters, num_candidates)
