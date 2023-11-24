import logging
import os

from validation.ordinal.impartial import (
    impartial_observed_frequencies,
    impartial_distribution,
)
from validation.ordinal.mallows import (
    mallows_observed_frequencies,
    mallows_distribution,
    frequencies_by_distance,
)
from validation.ordinal.singlepeaked import (
    single_peaked_walsh_distribution,
    get_all_single_peaked_ranks,
    single_peaked_walsh_observed_frequencies,
    single_peaked_contizer_observed_frequencies,
    single_peaked_conitzer_distribution,
)
from validation.utils import get_all_ranks
from validation.validators import run_chi_square_test, plot_frequencies


def run_mallows_validator(num_observations, num_candidates, all_ranks, plot_dir=None):
    central_rank = tuple(range(num_candidates))
    phi = 0.5

    frequencies = mallows_observed_frequencies(
        central_rank, phi, num_observations, all_ranks
    )
    distribution = mallows_distribution(central_rank, phi, all_ranks)

    logging.info("========================")
    logging.info("Mallow's model validator")
    logging.info("========================")
    test_result = run_chi_square_test(frequencies, distribution)

    plot_file_path = None
    if plot_dir:
        plot_file_path = os.path.join(plot_dir, f"Frequencies_Mallows_{phi}.pdf")
    plot_frequencies(
        frequencies,
        distribution,
        graph_sub_title=f"Mallow's model with phi={phi}\n"
        f"(#sample = {num_observations}, p-value={test_result.pvalue})",
        file_path=plot_file_path,
    )

    freq_by_dist = frequencies_by_distance(frequencies, central_rank, all_ranks)
    distrib_by_dist = frequencies_by_distance(distribution, central_rank, all_ranks)
    if plot_dir:
        plot_file_path = os.path.join(
            plot_dir, f"Frequencies_Mallows_{phi}_distance.pdf"
        )
    plot_frequencies(
        freq_by_dist,
        distrib_by_dist,
        graph_sub_title=f"Mallow's model with phi={phi}",
        xlabel="Distance to central ranking",
        file_path=plot_file_path,
    )


def run_impartial_ordinal_validator(num_observations, all_ranks, plot_dir=None):
    frequencies = impartial_observed_frequencies(num_observations, all_ranks)
    distribution = impartial_distribution(all_ranks)

    logging.info("========================")
    logging.info("Ordinal impartial cultures validator")
    logging.info("========================")
    test_result = run_chi_square_test(frequencies, distribution)

    plot_file_path = None
    if plot_dir:
        plot_file_path = os.path.join(plot_dir, f"Frequencies_Impartial.pdf")
    plot_frequencies(
        frequencies,
        distribution,
        graph_sub_title=f"Ordinal impartial culture\n"
        f"(#sample = {num_observations}, p-value={test_result.pvalue})",
        file_path=plot_file_path,
    )


def run_single_peaked_validator(num_observations, num_candidates, plot_dir=None):
    all_sp_ranks = get_all_single_peaked_ranks(num_candidates)
    frequencies = single_peaked_walsh_observed_frequencies(
        num_observations, all_sp_ranks
    )
    distribution = single_peaked_walsh_distribution(all_sp_ranks)

    logging.info("========================")
    logging.info("Single-peaked Walsh validator")
    logging.info("========================")
    test_result = run_chi_square_test(frequencies, distribution)

    plot_file_path = None
    if plot_dir:
        plot_file_path = os.path.join(plot_dir, f"Frequencies_SP_Walsh.pdf")
    plot_frequencies(
        frequencies,
        distribution,
        graph_sub_title=f"Single-peaked Walsh\n"
        f"(#sample = {num_observations}, p-value={test_result.pvalue})",
        file_path=plot_file_path,
        xlabel="Single-peaked rank identifier (ordered by theoretical frequency)",
    )

    frequencies = single_peaked_contizer_observed_frequencies(
        num_observations, all_sp_ranks
    )
    distribution = single_peaked_conitzer_distribution(all_sp_ranks)

    logging.info("========================")
    logging.info("Single-peaked Conitzer validator")
    logging.info("========================")
    test_result = run_chi_square_test(frequencies, distribution)

    plot_file_path = None
    if plot_dir:
        plot_file_path = os.path.join(plot_dir, f"Frequencies_SP_Conitzer.pdf")
    plot_frequencies(
        frequencies,
        distribution,
        graph_sub_title=f"Single-peaked Conitzer\n"
        f"(#sample = {num_observations}, p-value={test_result.pvalue})",
        file_path=plot_file_path,
        xlabel="Single-peaked rank identifier (ordered by theoretical frequency)",
        x_tick_labels=[" < ".join([str(j) for j in rank]) for rank in all_sp_ranks]
    )


def run_all_validators():
    num_candidates = 4
    num_observations = 100000

    all_ranks = get_all_ranks(num_candidates)

    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    plot_dir_root = os.path.join("plots", "ordinal")
    os.makedirs(plot_dir_root, exist_ok=True)

    mallows_dir = os.path.join(plot_dir_root, "mallows")
    os.makedirs(mallows_dir, exist_ok=True)
    run_mallows_validator(
        num_observations, num_candidates, all_ranks, plot_dir=mallows_dir
    )

    impartial_dir = os.path.join(plot_dir_root, "impartial")
    os.makedirs(impartial_dir, exist_ok=True)
    run_impartial_ordinal_validator(num_observations, all_ranks, plot_dir=impartial_dir)

    single_peaked_dir = os.path.join(plot_dir_root, "single_peaked")
    os.makedirs(single_peaked_dir, exist_ok=True)
    run_single_peaked_validator(
        num_observations, num_candidates, plot_dir=single_peaked_dir
    )


if __name__ == "__main__":
    run_all_validators()
