from prefsampling.ordinal import identity
from tests.utils import TestSampler


def all_test_samplers_ordinal_identity():
    return [TestSampler(identity, {})]
