from unittest import TestCase

from prefsampling import TreeSampler
from prefsampling.core import mixture
from prefsampling.ordinal import group_separable


def random_ord_group_separable_samplers():
    return [
        lambda num_voters, num_candidates, seed=None: group_separable(
            num_voters, num_candidates, tree_sampler, seed=seed
        )
        for tree_sampler in [TreeSampler.SCHROEDER, TreeSampler.SCHROEDER_UNIFORM, TreeSampler.SCHROEDER_LESCANNE]
    ]


class TestOrdinalGroupSeparable(TestCase):
    def test_ordinal_group_separable(self):
        with self.assertRaises(ValueError):
            group_separable(4, 5, "caterpillar")

        group_separable(3, 20, TreeSampler.SCHROEDER)

        mixture(
            4,
            20,
            [group_separable, group_separable, group_separable],
            [0.5, 0.2, 0.3],
            [{"tree_sampler": TreeSampler.SCHROEDER}, {"tree_sampler": TreeSampler.SCHROEDER}, {"tree_sampler": TreeSampler.SCHROEDER}],
        )