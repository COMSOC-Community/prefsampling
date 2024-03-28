"""
Module for sampling approval preferences: preferences in which a voter either approves or
disapprove each candidate.
"""

from prefsampling.approval.impartial import impartial, impartial_constant_size
from prefsampling.approval.identity import identity, full, empty
from prefsampling.approval.resampling import (
    resampling,
    disjoint_resampling,
    moving_resampling,
)
from prefsampling.approval.noise import noise, NoiseType
from prefsampling.approval.euclidean import (
    euclidean_threshold,
    euclidean_vcr,
    euclidean_constant_size,
)
from prefsampling.approval.truncated_ordinal import truncated_ordinal
from prefsampling.approval.urn import urn, urn_constant_size, urn_partylist


__all__ = [
    "impartial",
    "impartial_constant_size",
    "identity",
    "full",
    "empty",
    "resampling",
    "disjoint_resampling",
    "moving_resampling",
    "noise",
    "NoiseType",
    "euclidean_threshold",
    "euclidean_vcr",
    "euclidean_constant_size",
    "urn_partylist",
    "truncated_ordinal",
    "urn",
    "urn_constant_size",
]
