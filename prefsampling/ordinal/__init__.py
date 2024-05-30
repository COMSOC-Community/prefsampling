"""
Module for sampling ordinal preferences: preferences in which a voter ranks the candidates by
order of preference.
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
from prefsampling.ordinal.singlecrossing import (
    single_crossing,
    single_crossing_impartial,
)
from prefsampling.ordinal.mallows import (
    mallows,
    norm_mallows,
)
from prefsampling.ordinal.euclidean import euclidean, EuclideanSpace
from prefsampling.ordinal.plackettluce import plackett_luce
from prefsampling.ordinal.groupseparable import group_separable, TreeSampler
from prefsampling.ordinal.identity import identity
from prefsampling.ordinal.didi import didi
from prefsampling.ordinal.identity import identity

__all__ = [
    "urn",
    "impartial",
    "impartial_anonymous",
    "stratification",
    "single_peaked_walsh",
    "single_peaked_conitzer",
    "single_peaked_circle",
    "single_crossing",
    "single_crossing_impartial",
    "mallows",
    "norm_mallows",
    "euclidean",
    "EuclideanSpace",
    "plackett_luce",
    "group_separable",
    "TreeSampler",
    "identity",
    "didi",
]
