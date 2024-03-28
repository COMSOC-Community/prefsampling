from unittest import TestCase

from prefsampling.ordinal.singlepeaked import (
    single_peaked_walsh,
    single_peaked_conitzer,
    single_peaked_circle,
)
from tests.utils import TestSampler


def all_test_samplers_ordinal_single_peaked():
    return [
        TestSampler(single_peaked_conitzer, {}),
        TestSampler(single_peaked_circle, {}),
        TestSampler(single_peaked_walsh, {}),
    ]
