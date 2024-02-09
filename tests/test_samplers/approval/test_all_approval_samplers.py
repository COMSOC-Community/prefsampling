from unittest import TestCase


from prefsampling.approval import (
    resampling,
    disjoint_resampling,
    moving_resampling,
    impartial,
    euclidean,
    noise,
    identity,
    full,
    empty,
    urn_partylist,
    NoiseType,
    truncated_ordinal,
)
from prefsampling.core import resample_as_central_vote, rename_candidates, permute_voters, mixture
from prefsampling.core.euclidean import EuclideanSpace
from prefsampling.ordinal import urn as ordinal_urn

ALL_APPROVAL_SAMPLERS = [
    lambda num_voters, num_candidates, seed=None: resampling(
        num_voters, num_candidates, 0.5, 0.5, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: disjoint_resampling(
        num_voters, num_candidates, 0.5, 0.5, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: moving_resampling(
        num_voters, num_candidates, 0.5, 0.5, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: impartial(
        num_voters, num_candidates, 0.5, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: euclidean(
        num_voters, num_candidates, space=EuclideanSpace.UNIFORM, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: euclidean(
        num_voters, num_candidates, space=EuclideanSpace.GAUSSIAN, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: euclidean(
        num_voters, num_candidates, space=EuclideanSpace.SPHERE, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: euclidean(
        num_voters, num_candidates, space=EuclideanSpace.BALL, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: noise(
        num_voters, num_candidates, 0.5, 0.5, noise_type=NoiseType.HAMMING, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: noise(
        num_voters, num_candidates, 0.5, 0.5, noise_type=NoiseType.ZELINKA, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: noise(
        num_voters, num_candidates, 0.5, 0.5, noise_type=NoiseType.JACCARD, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: noise(
        num_voters,
        num_candidates,
        0.5,
        0.5,
        noise_type=NoiseType.BUNKE_SHEARER,
        seed=seed,
    ),
    lambda num_voters, num_candidates, seed=None: identity(
        num_voters, num_candidates, 0.5, seed=seed
    ),
    full,
    empty,
    lambda num_voters, num_candidates, seed=None: urn_partylist(
        num_voters, num_candidates, 0.1, 3, seed=seed
    ),

    lambda num_voters, num_candidates, seed=None: resample_as_central_vote(
        identity(num_voters, num_candidates, p=0.5),
        resampling,
        {"phi": 0.4, "p": 0.523, "seed": seed, "num_candidates": num_candidates},
    ),
    lambda num_voters, num_candidates, seed=None: rename_candidates(
        identity(num_voters, num_candidates, p=0.5),
        seed=seed,
        num_candidates=num_candidates
    ),
    lambda num_voters, num_candidates, seed=None: permute_voters(
        identity(num_voters, num_candidates, p=0.5),
        seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: mixture(
        num_voters,
        num_candidates,
        [identity, full, urn_partylist],
        [0.5, 0.2, 0.3],
        [{"p": 0.4}, {}, {"alpha": 0.1, "parties": 3}],
    ),
    lambda num_voters, num_candidates, seed=None: truncated_ordinal(
        num_voters,
        num_candidates,
        0.5,
        ordinal_urn,
        {"alpha": 0.8}
    ),
]


class TestApprovalSamplers(TestCase):
    def helper_test_all_approval_samplers(self, sampler, num_voters, num_candidates):
        result = sampler(num_voters, num_candidates)

        # Test whether the function returns a list of the correct size
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) == num_voters)

        # Test whether we have sets
        for vote in result:
            self.assertIsInstance(vote, set)

        # Test whether the values are within the range of candidates
        for vote in result:
            self.assertTrue(vote <= set(range(num_candidates)))

        # Test whether the value are int
        for vote in result:
            for candidate in vote:
                self.assertTrue(int(candidate) == candidate)

    def test_all_approval_samplers(self):
        num_voters = 200
        num_candidates = 5

        for sampler in ALL_APPROVAL_SAMPLERS:
            for test_sampler in [
                sampler,
                lambda x, y: sampler(num_voters=x, num_candidates=y),
                lambda x, y: sampler(x, y, seed=363),
            ]:
                with self.subTest(sampler=test_sampler):
                    self.helper_test_all_approval_samplers(
                        test_sampler, num_voters, num_candidates
                    )
