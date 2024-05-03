from unittest import TestCase

from prefsampling.ordinal.impartial import (
    impartial_theoretical_distribution as ord_impartial_distrib,
    stratification_theoretical_distribution as ord_strat_distrib,
    impartial_anonymous_theoretical_distribution as ord_impartial_anon_ditrib,
)
from prefsampling.ordinal.plackettluce import (
    theoretical_distribution as ord_plackett_distrib,
)
from prefsampling.ordinal.singlepeaked import (
    conitzer_theoretical_distribution as ord_sp_con_distrib,
    walsh_theoretical_distribution as ord_sp_wal_distrib,
    circle_theoretical_distribution as ord_sp_circ_distrib,
)
from prefsampling.ordinal.singlecrossing import (
    impartial_theoretical_distribution as ord_sc_impartial_distrib,
)
from prefsampling.ordinal.urn import theoretical_distribution as ord_urn_distrib
from prefsampling.ordinal.mallows import theoretical_distribution as ord_mallows_distrib

from prefsampling.approval.resampling import (
    resampling_theoretical_distribution,
    disjoint_resampling_theoretical_distribution,
)
from prefsampling.approval.noise import (
    theoretical_distribution as app_noise_distrib,
    SetDistance,
)


class TestInputValidators(TestCase):
    def test_all_theoretical_distribution(self):
        ord_impartial_distrib(2)
        with self.assertRaises(ValueError):
            ord_impartial_distrib()
        ord_impartial_anon_ditrib(2, 3)
        with self.assertRaises(ValueError):
            ord_impartial_anon_ditrib()
        with self.assertRaises(ValueError):
            ord_impartial_anon_ditrib(num_candidates=3)
        ord_strat_distrib(2, 0.58)
        ord_plackett_distrib([0.2, 0.3], 2)
        with self.assertRaises(ValueError):
            ord_plackett_distrib([0.2, 0.3])
        ord_sp_con_distrib(3)
        ord_sp_wal_distrib(3)
        with self.assertRaises(ValueError):
            ord_sp_wal_distrib()
        ord_sp_circ_distrib(3)
        with self.assertRaises(ValueError):
            ord_sp_circ_distrib()
        ord_sc_impartial_distrib(2, 3)
        with self.assertRaises(ValueError):
            ord_sc_impartial_distrib()
        with self.assertRaises(ValueError):
            ord_sc_impartial_distrib(num_candidates=2)
        ord_urn_distrib(2, 3, 0.5)
        ord_mallows_distrib(3, 0.3, normalise_phi=True)
        ord_mallows_distrib(3, 0.3, normalise_phi=False)

        resampling_theoretical_distribution(3, 0.5, 0.2)
        disjoint_resampling_theoretical_distribution(3, 0.3, 0.4, 2)
        app_noise_distrib(3, 0.7, SetDistance.JACCARD, 0.6)
        with self.assertRaises(ValueError):
            app_noise_distrib(3, 0.7, "aze", 0.6)
