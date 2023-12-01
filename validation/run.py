import logging
import os

import numpy as np

from validation.ordinal.impartial import OrdinalImpartialValidator
from validation.ordinal.mallows import OrdinalMallowsValidator
from validation.ordinal.plackettluce import PlackettLuceValidator
from validation.ordinal.single_crossing import SingleCrossingValidator
from validation.ordinal.singlepeaked import SPWalshValidator, SPConitzerValidator
from validation.utils import get_all_ranks, get_all_single_peaked_ranks, \
    get_all_single_crossing_profiles, get_all_profiles


def run_ordinal_iid_validators(num_observations):
    num_candidates = 5

    all_ranks = get_all_ranks(num_candidates)

    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    plot_dir_root = os.path.join("plots", "ordinal")
    os.makedirs(plot_dir_root, exist_ok=True)

    mallows_dir = os.path.join(plot_dir_root, "mallows")
    os.makedirs(mallows_dir, exist_ok=True)
    phi = 0.5
    logging.info("==================")
    logging.info("Mallow's validator")
    logging.info("==================")
    mallows_validator = OrdinalMallowsValidator(
        num_candidates, phi, tuple(range(num_candidates)), all_outcomes=all_ranks
    )
    mallows_validator.run(
        num_observations,
        model_name="Mallows' model",
        graph_file_path=os.path.join(mallows_dir, f"Frequencies_Mallows_{phi}.png"),
    )

    impartial_dir = os.path.join(plot_dir_root, "impartial")
    os.makedirs(impartial_dir, exist_ok=True)
    logging.info("===================")
    logging.info("Impartial validator")
    logging.info("===================")
    impartial_validator = OrdinalImpartialValidator(
        num_candidates, all_outcomes=all_ranks
    )
    impartial_validator.run(
        num_observations,
        model_name="impartial culture",
        graph_file_path=os.path.join(impartial_dir, f"Frequencies_Impartial.png"),
    )

    single_peaked_dir = os.path.join(plot_dir_root, "single_peaked")
    os.makedirs(single_peaked_dir, exist_ok=True)
    all_sp_ranks = get_all_single_peaked_ranks(num_candidates)
    logging.info("=============================")
    logging.info("Single-peaked Walsh validator")
    logging.info("=============================")
    sp_walsh_validator = SPWalshValidator(num_candidates, all_outcomes=all_sp_ranks)
    sp_walsh_validator.run(
        num_observations,
        model_name="single peaked Walsh' model",
        graph_file_path=os.path.join(single_peaked_dir, f"Frequencies_SP_Walsh.png"),
    )
    logging.info("================================")
    logging.info("Single-peaked Conitzer validator")
    logging.info("================================")
    sp_conitzer_validator = SPConitzerValidator(
        num_candidates, all_outcomes=all_sp_ranks
    )
    sp_conitzer_validator.run(
        num_observations,
        model_name="single peaked Walsh' model",
        graph_file_path=os.path.join(single_peaked_dir, f"Frequencies_SP_Conitzer.png"),
    )

    plackettluce_dir = os.path.join(plot_dir_root, "plackettluce")
    os.makedirs(plackettluce_dir, exist_ok=True)
    logging.info("=======================")
    logging.info("Plackett Luce validator")
    logging.info("=======================")
    plakettluce_validator = PlackettLuceValidator(
        num_candidates, alphas=[1] * num_candidates, all_outcomes=all_ranks
    )
    plakettluce_validator.run(
        num_observations,
        model_name="Plackett Luce",
        graph_file_path=os.path.join(plackettluce_dir, f"Frequencies_Plackett_Luce_1s.png"),
    )
    plakettluce_validator = PlackettLuceValidator(
        num_candidates, alphas=np.random.random(num_candidates), all_outcomes=all_ranks
    )
    plakettluce_validator.run(
        num_observations,
        model_name="Plackett Luce",
        graph_file_path=os.path.join(plackettluce_dir, f"Frequencies_Plackett_Luce_rnd.png"),
    )


def run_ordinal_non_iid_validators(num_observations):
    num_candidates = 3
    num_voters = 1

    all_profiles = get_all_profiles(num_voters, num_candidates)

    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    plot_dir_root = os.path.join("plots", "ordinal")
    os.makedirs(plot_dir_root, exist_ok=True)

    all_sc_profiles = get_all_single_crossing_profiles(num_voters,
                                                       num_candidates,
                                                       all_profiles=all_profiles)
    singlecrossing_dir = os.path.join(plot_dir_root, "singlecrossing")
    os.makedirs(singlecrossing_dir, exist_ok=True)
    logging.info("=========================")
    logging.info("Single Crossing validator")
    logging.info("=========================")
    singlecrossing_validator = SingleCrossingValidator(
        num_voters,
        num_candidates,
        all_outcomes=all_sc_profiles
    )
    singlecrossing_validator.run(
        num_observations,
        model_name="Single Crossing",
        graph_file_path=os.path.join(singlecrossing_dir, f"Frequencies_single_crosssing.png"),
        graph_xlabel="Profile identifier (ordered by theoretical frequency)",
        graph_x_tick_labels=all_sc_profiles
    )


if __name__ == "__main__":
    num_observations = 1000000

    # run_ordinal_iid_validators(num_observations)
    run_ordinal_non_iid_validators(num_observations)
