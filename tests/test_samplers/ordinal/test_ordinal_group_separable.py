from enum import Enum
from unittest import TestCase

from prefsampling import TreeSampler
from prefsampling.core import mixture
from prefsampling.ordinal import group_separable
from tests.utils import TestSampler


def all_test_samplers_ordinal_group_separable():
    return [
        TestSampler(group_separable, {"tree_sampler": tree_sampler})
        for tree_sampler in [
            TreeSampler.SCHROEDER,
            TreeSampler.SCHROEDER_UNIFORM,
            TreeSampler.SCHROEDER_LESCANNE,
        ]
    ]


class TestOrdinalGroupSeparable(TestCase):
    def test_ordinal_group_separable(self):
        with self.assertRaises(ValueError):
            group_separable(4, 5, "caterpillar")
        with self.assertRaises(ValueError):
            class TestEnum(Enum):
                a = "1"
            group_separable(4, 5, TestEnum.a)


        group_separable(3, 20, TreeSampler.SCHROEDER)

        mixture(
            4,
            20,
            [group_separable, group_separable, group_separable],
            [0.5, 0.2, 0.3],
            [
                {"tree_sampler": TreeSampler.SCHROEDER},
                {"tree_sampler": TreeSampler.SCHROEDER},
                {"tree_sampler": TreeSampler.SCHROEDER},
            ],
        )
