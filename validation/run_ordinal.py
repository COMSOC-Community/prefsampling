import logging
import os

from validation.ordinal.impartial import OrdinalImpartialValidator
from validation.ordinal.mallows import OrdinalMallowsValidator
from validation.ordinal.singlepeaked import SPWalshValidator, SPConitzerValidator
from validation.utils import get_all_ranks, get_all_single_peaked_ranks


def run_all_validators():
    num_candidates = 5
    num_observations = 100000

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


if __name__ == "__main__":
    run_all_validators()
