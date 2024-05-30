import math
from unittest import TestCase

from prefsampling.combinatorics import (
    comb,
    _comb,
    generalised_ascending_factorial,
    powerset,
    proper_powerset,
    all_rankings,
    all_anonymous_profiles,
    all_profiles,
    all_non_isomorphic_profiles,
    all_single_peaked_rankings,
    all_single_peaked_circle_rankings,
    is_single_crossing,
    all_single_crossing_profiles,
    all_group_separable_profiles,
    all_gs_structure,
    kendall_tau_distance,
    gs_structure,
    GSNode,
)


class TestCombinatorics(TestCase):

    def test_comb(self):
        for c in [comb, _comb]:
            self.assertEqual(c(0, 0), 1)
            self.assertEqual(c(0, 1), 0)
            self.assertEqual(c(0, 2), 0)
            self.assertEqual(c(0, 3), 0)
            self.assertEqual(c(0, 4), 0)
            self.assertEqual(c(1, 0), 1)
            self.assertEqual(c(1, 1), 1)
            self.assertEqual(c(1, 2), 0)
            self.assertEqual(c(1, 3), 0)
            self.assertEqual(c(1, 4), 0)
            self.assertEqual(c(2, 0), 1)
            self.assertEqual(c(2, 1), 2)
            self.assertEqual(c(2, 2), 1)
            self.assertEqual(c(2, 3), 0)
            self.assertEqual(c(2, 4), 0)
            self.assertEqual(c(3, 0), 1)
            self.assertEqual(c(3, 1), 3)
            self.assertEqual(c(3, 2), 3)
            self.assertEqual(c(3, 3), 1)
            self.assertEqual(c(3, 4), 0)
            self.assertEqual(c(4, 0), 1)
            self.assertEqual(c(4, 1), 4)
            self.assertEqual(c(4, 2), 6)
            self.assertEqual(c(4, 3), 4)
            self.assertEqual(c(4, 4), 1)

    def test_generalised_ascending_factorial(self):
        for x in range(8):
            self.assertEqual(generalised_ascending_factorial(x, 0, 1), 1)
            self.assertEqual(generalised_ascending_factorial(x, 1, 1), x)
            self.assertEqual(generalised_ascending_factorial(x, 2, 1), x**2 + x)
            self.assertEqual(
                generalised_ascending_factorial(x, 3, 1), x**3 + 3 * x**2 + 2 * x
            )
            self.assertEqual(
                generalised_ascending_factorial(x, 4, 1),
                x**4 + 6 * x**3 + 11 * x**2 + 6 * x,
            )

    def test_powerset(self):
        for x in range(1, 8):
            self.assertEqual(len(powerset(range(x), min_size=0)), 2**x)
            self.assertEqual(len(powerset(range(x), min_size=1)), 2**x - 1)
            self.assertEqual(len(powerset(range(x), min_size=1, max_size=x)), 2**x - 2)
            self.assertEqual(len(proper_powerset(range(x), min_size=0)), 2**x - 1)
            self.assertEqual(len(proper_powerset(range(x), min_size=1)), 2**x - 2)

    def test_all_rankings(self):
        for x in range(1, 8):
            self.assertEqual(len(set(all_rankings(x))), math.factorial(x))

    def test_kendall_tau_distance(self):
        self.assertEqual(kendall_tau_distance((0, 1, 2, 3), (0, 1, 3, 2)), 1)
        self.assertEqual(kendall_tau_distance((0, 1, 2, 3), (0, 1, 2, 3)), 0)
        self.assertEqual(kendall_tau_distance((0, 1, 2, 3), (3, 2, 1, 0)), 6)

    def test_all_the_rest(self):
        all_anonymous_profiles(3, 4)
        all_profiles(3, 4)
        all_non_isomorphic_profiles(3, 4)
        all_single_peaked_rankings(5)
        all_single_peaked_circle_rankings(5)
        is_single_crossing(((0, 1, 2, 3), (3, 2, 1, 0)))
        all_single_crossing_profiles(3, 4, fix_order=True)
        all_single_crossing_profiles(3, 4, fix_order=False)
        all_group_separable_profiles(3, 4)
        all_gs_structure(3, 4)
        with self.assertRaises(ValueError):
            all_gs_structure()
        gs_structure(((0, 1, 2, 3), (3, 2, 1, 0)))
        gs_structure(((0, 1, 2, 3), (3, 2, 1, 0)), verbose=True)
        with self.assertRaises(ValueError):
            gs_structure(((0, 1, 2), (2, 0, 1), (1, 2, 0)), verbose=True)
        node = GSNode({0, 1, 2})
        node.__repr__()
        node.print_tree()
        node.tree_representation()
