from unittest import TestCase

import numpy as np

from prefsampling.approval.urn import urn, urn_constant_size, urn_partylist
from tests.utils import float_parameter_test_values


def random_app_urn_samplers():
    samplers = [
        lambda num_voters, num_candidates, seed=None: urn(
            num_voters, num_candidates, random_p, random_alpha, seed=seed
        )
        for random_p in float_parameter_test_values(0, 1, 4)
        for random_alpha in float_parameter_test_values(0, 10, 4)
    ]
    samplers += [
        lambda num_voters, num_candidates, seed=None: urn_constant_size(
            num_voters, num_candidates, random_p, random_alpha, seed=seed
        )
        for random_p in float_parameter_test_values(0, 1, 4)
        for random_alpha in float_parameter_test_values(0, 10, 4)
    ]
    samplers += [
        lambda num_voters, num_candidates, seed=None: urn_partylist(
            num_voters, num_candidates, random_p, random_num_parties, seed=seed
        )
        for random_p in float_parameter_test_values(0, 1, 4)
        for random_num_parties in range(1, 6)
    ]
    return samplers
