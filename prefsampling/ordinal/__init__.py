"""
Module for sampling ordinal preferences: preferences in which an agent ranks the candidates by order of preference.
"""

from prefsampling.ordinal.urn import urn
from prefsampling.ordinal.impartial import impartial_culture, impartial_anonymous_culture
from prefsampling.ordinal.singlepeaked import single_peaked_Conitzer, single_peaked_circle_Conitzer, single_peaked_Walsh
from prefsampling.ordinal.singlecrossing import single_crossing

__all__ = [
    "urn",
    "impartial_culture",
    "impartial_anonymous_culture",
    "single_peaked_Walsh",
    "single_peaked_Conitzer",
    "single_peaked_circle_Conitzer",
    "single_crossing"
]
