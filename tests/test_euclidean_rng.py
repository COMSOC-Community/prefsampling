from unittest import TestCase

import numpy as np

from prefsampling.core.euclidean import EuclideanSpace, sample_election_positions
from prefsampling.point import ball_resampling, cube, gaussian


class TestEuclideanStreams(TestCase):
    def test_spaces_are_reproducible_and_distinct(self):
        for space in EuclideanSpace:
            for counts in [(4, 4), (5, 3)]:
                for seed in [1, 42, 12345]:
                    with self.subTest(space=space, counts=counts, seed=seed):
                        args = (*counts, 2, space, space)
                        voters, candidates = sample_election_positions(*args, seed=seed)
                        repeated = sample_election_positions(*args, seed=seed)
                        np.testing.assert_array_equal(voters, repeated[0])
                        np.testing.assert_array_equal(candidates, repeated[1])
                        self.assertEqual(len(np.unique(voters, axis=0)), counts[0])
                        self.assertEqual(len(np.unique(candidates, axis=0)), counts[1])
                        self.assertFalse(np.array_equal(voters[:counts[1]], candidates))
                        if space == EuclideanSpace.GAUSSIAN_BALL:
                            self.assertTrue(np.all(np.linalg.norm(voters, axis=1) <= 0.5))
                            self.assertTrue(np.all(np.linalg.norm(candidates, axis=1) <= 0.5))

    def test_role_streams_do_not_depend_on_other_count_or_explicit_points(self):
        for space in EuclideanSpace:
            with self.subTest(space=space):
                voters, candidates = sample_election_positions(4, 3, 2, space, space, seed=42)
                other_voters = sample_election_positions(4, 7, 2, space, space, seed=42)[0]
                other_candidates = sample_election_positions(9, 3, 2, space, space, seed=42)[1]
                np.testing.assert_array_equal(voters, other_voters)
                np.testing.assert_array_equal(candidates, other_candidates)
                explicit_voters, sampled_candidates = sample_election_positions(
                    4, 3, 2, voters, space, seed=42
                )
                sampled_voters, explicit_candidates = sample_election_positions(
                    4, 3, 2, space, candidates, seed=42
                )
                np.testing.assert_array_equal(explicit_voters, voters)
                np.testing.assert_array_equal(sampled_candidates, candidates)
                np.testing.assert_array_equal(sampled_voters, voters)
                np.testing.assert_array_equal(explicit_candidates, candidates)

    def test_custom_sampler_seeds_and_shared_arguments(self):
        seeds = []

        def sampler(num_points, num_dimensions, seed, widths):
            seeds.append(seed)
            return cube(num_points, num_dimensions, widths=widths, seed=seed)

        shared = {"widths": 2, "seed": 17}
        sample_election_positions(4, 3, 2, sampler, sampler, shared, shared, seed=42)
        self.assertEqual(shared, {"widths": 2, "seed": 17})
        self.assertTrue(all(type(seed) is int for seed in seeds))
        self.assertNotEqual(seeds[0], seeds[1])
        seeds.clear()
        sample_election_positions(4, 3, 2, sampler, sampler, shared, shared)
        self.assertEqual(seeds, [None, None])
        self.assertEqual(shared, {"widths": 2, "seed": 17})

    def test_builtin_arguments_are_preserved(self):
        for space in EuclideanSpace:
            with self.subTest(space=space):
                shared = {"seed": 17}
                sample_election_positions(4, 3, 2, space, space, shared, shared, seed=42)
                self.assertEqual(shared, {"seed": 17})


class TestBallProposalStreams(TestCase):
    def test_each_proposal_gets_a_new_seed(self):
        for reject in [False, True]:
            with self.subTest(reject=reject):
                seeds = []

                def sampler(seed):
                    seeds.append(seed)
                    return [2, 0] if reject and len(seeds) % 2 else [0, 0]

                arguments = {"seed": 99}
                result = ball_resampling(4, 2, sampler, arguments, seed=42)
                expected_calls = 8 if reject else 4
                self.assertEqual(len(seeds), expected_calls)
                self.assertEqual(len(set(seeds)), expected_calls)
                self.assertTrue(all(type(seed) is int for seed in seeds))
                first_seeds = seeds.copy()
                seeds.clear()
                np.testing.assert_array_equal(
                    result, ball_resampling(4, 2, sampler, arguments, seed=42)
                )
                self.assertEqual(seeds, first_seeds)
                self.assertEqual(arguments, {"seed": 99})

    def test_inner_seed_is_used_when_outer_seed_is_none(self):
        def sampler(seed):
            return gaussian(1, 2, sigmas=0.1, seed=seed)[0]

        arguments = {"seed": 42}
        result = ball_resampling(5, 2, sampler, arguments)
        np.testing.assert_array_equal(result, ball_resampling(5, 2, sampler, arguments))
        self.assertEqual(len(np.unique(result, axis=0)), 5)
        self.assertEqual(arguments, {"seed": 42})

    def test_unseeded_sampler_does_not_require_seed_keyword(self):
        result = ball_resampling(3, 1, lambda: 0.1, {})
        np.testing.assert_array_equal(result, [[0.1], [0.1], [0.1]])
        seeds = []

        def sampler(seed):
            seeds.append(seed)
            return [0, 0]

        ball_resampling(3, 2, sampler, {"seed": None})
        self.assertEqual(seeds, [None, None, None])

    def test_retry_shapes_are_validated(self):
        for malformed in [[0], [[0], [0]]]:
            with self.subTest(malformed=malformed):
                proposals = iter([[2, 0], malformed])
                with self.assertRaises(ValueError):
                    ball_resampling(1, 2, lambda: next(proposals), {})

    def test_retry_limit_warns_and_uses_center(self):
        seeds = []

        def sampler(seed):
            seeds.append(seed)
            return [10, 10]

        with self.assertWarns(RuntimeWarning):
            result = ball_resampling(
                2, 2, sampler, {}, center_point=[1, 1], max_numer_resampling=2, seed=42
            )
        np.testing.assert_array_equal(result, [[1, 1], [1, 1]])
        self.assertEqual(len(seeds), 6)
        self.assertEqual(len(set(seeds)), 6)
