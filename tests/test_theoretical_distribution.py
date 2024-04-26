from unittest import TestCase

from prefsampling.ordinal.impartial import (
    impartial_theoretical_distribution as ord_impartial_distrib,
    stratification_theoretical_distribution as ord_strat_distrib,
    impartial_anonymous_theoretical_distribution as ord_impartial_anon_ditrib
)
from prefsampling.ordinal.plackettluce import theoretical_distribution as ord_plackett_distrib
from prefsampling.ordinal.singlepeaked import (
    conitzer_theoretical_distribution as ord_sp_con_distrib,
    walsh_theoretical_distribution as ord_sp_wal_distrib,
    circle_theoretical_distribution as ord_sp_circ_distrib,
)
from prefsampling.ordinal.urn import theoretical_distribution as ord_urn_distrib

from prefsampling.approval.resampling import resampling_theoretical_distribution, disjoint_resampling_theoretical_distribution


class TestInputValidators(TestCase):
    def test_all_theoretical_distribution(self):
        ord_impartial_distrib(2)
        ord_impartial_anon_ditrib(2, 3)
        ord_strat_distrib(2, 0.58)
        ord_plackett_distrib([0.2, 0.3], 2)
        with self.assertRaises(ValueError):
            ord_plackett_distrib([0.2, 0.3])
        ord_sp_con_distrib(3)
        ord_sp_wal_distrib(3)
        ord_sp_circ_distrib(3)
        ord_urn_distrib(2, 3, 0.5)

        resampling_theoretical_distribution(3, 0.5, 0.2)
        disjoint_resampling_theoretical_distribution(3, 0.3, 0.4, 2)
