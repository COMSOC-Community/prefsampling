from unittest import TestCase

import numpy as np

from prefsampling.inputvalidators import validate_num_voters_candidates
from prefsampling.ordinal.singlepeaked import (
    single_peaked_walsh,
    single_peaked_conitzer,
    single_peaked_circle,
)
from tests.utils import TestSampler


def all_test_samplers_ordinal_single_peaked():
    @validate_num_voters_candidates
    def single_peaked_conitzer_axis(num_voters, num_candidates, seed=None):
        return single_peaked_conitzer(
            num_voters,
            num_candidates,
            axis=list(np.random.permutation(num_candidates)),
            seed=seed,
        )

    @validate_num_voters_candidates
    def single_peaked_walsh_axis(num_voters, num_candidates, seed=None):
        return single_peaked_conitzer(
            num_voters,
            num_candidates,
            axis=list(np.random.permutation(num_candidates)),
            seed=seed,
        )

    @validate_num_voters_candidates
    def single_peaked_circle_axis(num_voters, num_candidates, seed=None):
        return single_peaked_conitzer(
            num_voters,
            num_candidates,
            axis=list(np.random.permutation(num_candidates)),
            seed=seed,
        )

    return [
        TestSampler(single_peaked_conitzer, {}),
        TestSampler(single_peaked_conitzer_axis, {}),
        TestSampler(single_peaked_conitzer_axis, {}),
        TestSampler(single_peaked_circle, {}),
        TestSampler(single_peaked_circle_axis, {}),
        TestSampler(single_peaked_circle_axis, {}),
        TestSampler(single_peaked_walsh, {}),
        TestSampler(single_peaked_walsh_axis, {}),
        TestSampler(single_peaked_walsh_axis, {}),
    ]


class TestSinglePeaked(TestCase):
    def test_axis(self):
        with self.assertRaises(ValueError):
            single_peaked_conitzer(2, 4, axis=[])
        with self.assertRaises(ValueError):
            single_peaked_conitzer(2, 4, axis=[0, 1, 2, 3, 4, 5])
        with self.assertRaises(ValueError):
            single_peaked_conitzer(2, 4, axis=[0, 1, 2, 3, 5])

        with self.assertRaises(ValueError):
            single_peaked_walsh(2, 4, axis=[])
        with self.assertRaises(ValueError):
            single_peaked_walsh(2, 4, axis=[0, 1, 2, 3, 4, 5])
        with self.assertRaises(ValueError):
            single_peaked_walsh(2, 4, axis=[0, 1, 2, 3, 5])

        with self.assertRaises(ValueError):
            single_peaked_circle(2, 4, axis=[])
        with self.assertRaises(ValueError):
            single_peaked_circle(2, 4, axis=[0, 1, 2, 3, 4, 5])
        with self.assertRaises(ValueError):
            single_peaked_circle(2, 4, axis=[0, 1, 2, 3, 5])
