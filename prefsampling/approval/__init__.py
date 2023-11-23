"""
Module for sampling approval preferences:
    preferences in which an agent either approves or disapprove each candidate.
"""

from prefsampling.approval.impartial import impartial_culture
from prefsampling.approval.identity import identity
from prefsampling.approval.resampling import resampling, disjoint_resampling
from prefsampling.approval.noise import noise
from prefsampling.approval.euclidean import euclidean


__all__ = [
    "impartial_culture",
    "identity",
    "resampling",
    "disjoint_resampling",
    "noise",
    "euclidean",
]
