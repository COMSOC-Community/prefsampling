"""
Module for sampling approval preferences: preferences in which a voter either approves or
disapprove each candidate.
"""

from prefsampling.approval.impartial import impartial
from prefsampling.approval.identity import identity
from prefsampling.approval.resampling import (
    resampling,
    disjoint_resampling,
    moving_resampling,
)
from prefsampling.approval.noise import noise
from prefsampling.approval.euclidean import euclidean


__all__ = [
    "impartial",
    "identity",
    "resampling",
    "disjoint_resampling",
    "moving_resampling",
    "noise",
    "euclidean",
]
