from unittest import TestCase

from prefsampling.ordinal.urn import urn as ordinal_urn
from prefsampling.ordinal.impartial import (
    impartial as ordinal_impartial_culture,
    impartial_anonymous as ordinal_impartial_anonymous_culture,
    stratification as ordinal_stratification,
)
from prefsampling.ordinal.singlecrossing import (
    single_crossing as ordinal_single_crossing,
    single_crossing_impartial as ordinal_single_crossing_impartial,
)
from prefsampling.ordinal.singlepeaked import (
    single_peaked_conitzer as ordinal_single_peaked_conitzer,
    single_peaked_circle as ordinal_single_peaked_circle,
    single_peaked_walsh as ordinal_single_peaked_walsh,
)
from prefsampling.ordinal.euclidean import euclidean as ordinal_euclidean
from prefsampling.ordinal.mallows import (
    mallows as ordinal_mallows,
    norm_mallows as ordinal_norm_mallows,
)
from prefsampling.ordinal.plackettluce import plackett_luce as ordinal_plackett_luce
from prefsampling.ordinal.didi import didi as ordinal_didi
from prefsampling.ordinal.identity import identity as ordinal_identity

from prefsampling.approval.resampling import (
    resampling as approval_resampling,
    disjoint_resampling as approval_disjoint_resampling,
    moving_resampling as approval_moving_resampling,
)
from prefsampling.approval.impartial import (
    impartial as approval_impartial_culture,
)
from prefsampling.approval.euclidean import euclidean as approval_euclidean
from prefsampling.approval.noise import noise as approval_noise
from prefsampling.approval.identity import (
    identity as approval_identity,
    full as approval_full,
    empty as approval_empty,
)
from prefsampling.approval.partylist import urn_partylist as approval_urn_partylist

ALL_SAMPLERS = [
    ordinal_impartial_culture,
    ordinal_impartial_anonymous_culture,
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
    ordinal_euclidean,
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
    lambda num_voters, num_candidates, seed=None: approval_resampling(
        num_voters, num_candidates, 0.5, 0.5, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: approval_disjoint_resampling(
        num_voters, num_candidates, 0.5, 0.5, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: approval_moving_resampling(
        num_voters, num_candidates, 0.5, 0.5, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: approval_impartial_culture(
        num_voters, num_candidates, 0.5, seed=seed
    ),
    approval_euclidean,
    lambda num_voters, num_candidates, seed=None: approval_noise(
        num_voters, num_candidates, 0.5, 0.5, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: approval_identity(
        num_voters, num_candidates, 0.5, seed=seed
    ),
    approval_full,
    approval_empty,
    lambda num_voters, num_candidates, seed=None: approval_urn_partylist(
        num_voters, num_candidates, 0.1, 3, seed=seed
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
