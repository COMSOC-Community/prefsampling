"""
Module for sampling ordinal preferences:
    preferences in which an agent ranks the candidates by order of preference.
"""

from prefsampling.ordinal.urn import urn
from prefsampling.ordinal.impartial import (
    impartial,
    impartial_anonymous,
    stratification,
)
from prefsampling.ordinal.singlepeaked import (
    single_peaked_conitzer,
    single_peaked_circle,
    single_peaked_walsh,
)
from prefsampling.ordinal.singlecrossing import single_crossing
from prefsampling.ordinal.mallows import (
    mallows,
    norm_mallows,
)
from prefsampling.ordinal.euclidean import euclidean
from prefsampling.ordinal.plackettluce import plackett_luce

__all__ = [
    "urn",
    "impartial",
    "impartial_anonymous",
    "stratification",
    "single_peaked_walsh",
    "single_peaked_conitzer",
    "single_peaked_circle",
    "single_crossing",
    "mallows",
    "norm_mallows",
    "euclidean",
    "plackett_luce"
]
