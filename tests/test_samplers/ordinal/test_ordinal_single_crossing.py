from unittest import TestCase

from prefsampling.ordinal.singlecrossing import (
    SingleCrossingNode,
    single_crossing,
    single_crossing_impartial,
)
from tests.utils import TestSampler


def all_test_samplers_ordinal_single_crossing():
    return [
        TestSampler(single_crossing, {}),
        TestSampler(single_crossing_impartial, {}),
    ]


class TestOrdinalSingleCrossing(TestCase):
    def test_ordinal_single_crossing_node(self):
        n = SingleCrossingNode((0, 1, 2))
        n.__repr__()
        n.__str__()
        self.assertEqual(n, (0, 1, 2))
        self.assertNotEqual(n, (0, 2))
        self.assertNotEqual(n, [0, 1, 2])
        self.assertTrue(n.count_elections(0) == 0)
        self.assertTrue(n.count_elections(1) == 1)
        with self.assertRaises(ValueError):
            n.count_elections(-1)
