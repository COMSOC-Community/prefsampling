from unittest import TestCase

from prefsampling.ordinal.singlepeaked import single_peaked_walsh, single_peaked_conitzer, single_peaked_circle


def random_ord_single_peaked_samplers():
    return [
        single_peaked_conitzer,
        single_peaked_circle,
        single_peaked_walsh,
    ]

