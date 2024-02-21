__author__ = "Simon Rey and Stanisław Szufa"
__email__ = "reysimon@orange.fr"
__version__ = "0.1.8"

from enum import Enum
from itertools import chain

from prefsampling.approval import NoiseType
from prefsampling.core.euclidean import EuclideanSpace
from prefsampling.ordinal import TreeSampler


class CONSTANTS(Enum):
    """ All constants of the package """
    _ignore_ = 'member cls'
    cls = vars()
    for member in chain(list(EuclideanSpace), list(TreeSampler), list(NoiseType)):
        if member.name in cls:
            raise ValueError(f"The name {member.name} is used in more than one enumeration. The"
                             f"CONSTANTS class needs unique names to be well-defined.")
        cls[member.name] = member.value
