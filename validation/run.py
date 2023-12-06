import logging
import math
import os
from collections.abc import Iterable

import numpy as np

from prefsampling.core.euclidean import EuclideanSpace
from validation.ordinal.euclidean import OrdinalEuclideanValidator
from validation.ordinal.impartial import (
    OrdinalImpartialValidator,
    OrdinalImpartialAnonymousValidator,
    StratificationValidator,
)
from validation.ordinal.mallows import OrdinalMallowsValidator
from validation.ordinal.plackettluce import PlackettLuceValidator
from validation.ordinal.singlecrossing import (
    SingleCrossingValidator,
    SingleCrossingImpartialValidator,
)
from validation.ordinal.singlepeaked import (
    SPWalshValidator,
    SPConitzerValidator,
    SPCircleValidator,
)
from validation.ordinal.urn import UrnValidator
from validation.utils import (
    get_all_ranks,
    get_all_single_peaked_ranks,
    get_all_anonymous_profiles,
    get_all_single_peaked_circle_ranks,
    get_all_sc_profiles_non_iso,
)


def run_validator(
    long_model_name,
    short_model_name,
    validator,
    num_obs,
    plot_dir_root,
    graph_title="",
    graph_xlabel="",
    graph_ylabel="",
    graph_x_tick_labels=None,
    graph_ordering="theoretical-observed",
):
    logging.info("=" * len(long_model_name + " validator"))
    logging.info(f"{long_model_name} validator")
    logging.info("=" * len(long_model_name + " validator"))
    validator.run(
        num_obs,
        model_name=long_model_name,
        graph_title=graph_title,
        graph_xlabel=graph_xlabel,
        graph_ylabel=graph_ylabel,
        graph_x_tick_labels=graph_x_tick_labels,
        graph_file_path=os.path.join(plot_dir_root, f"{short_model_name}.png"),
        graph_ordering=graph_ordering,
    )


def run_ordinal_mallows_validator(num_obs, phi, all_ranks, plot_dir_root):
    if not isinstance(phi, Iterable):
        phi = [phi]
    for p in phi:
        validator = OrdinalMallowsValidator(
            len(all_ranks[0]),
            p,
            tuple(range(len(all_ranks[0]))),
            all_outcomes=all_ranks,
        )
        run_validator(
            "Mallows' model",
            f"mallows_{p}",
            validator,
            num_obs,
            plot_dir_root,
        )


def run_ordinal_impartial_validator(num_obs, all_ranks, plot_dir_root):
    validator = OrdinalImpartialValidator(len(all_ranks[0]), all_outcomes=all_ranks)
    run_validator(
        "Ordinal Impartial model",
        "ordinal_impartial",
        validator,
        num_obs,
        plot_dir_root,
    )


def run_stratification_validator(num_obs, weight, all_ranks, plot_dir_root):
    if not isinstance(weight, Iterable):
        weight = [weight]
    for w in weight:
        validator = StratificationValidator(
            len(all_ranks[0]), weight=w, all_outcomes=all_ranks
        )
        run_validator(
            "Stratification model",
            f"stratification_{w}",
            validator,
            num_obs,
            plot_dir_root,
        )


def run_ordinal_impartial_anonymous_validator(
    num_obs, all_anonymous_profiles, plot_dir_root
):
    validator = OrdinalImpartialAnonymousValidator(
        len(all_anonymous_profiles[0]),
        len(all_anonymous_profiles[0][0]),
        all_outcomes=all_anonymous_profiles,
    )
    run_validator(
        "Ordinal Impartial Anonymous model",
        "ordinal_impartial_anonymous",
        validator,
        num_obs,
        plot_dir_root,
        graph_xlabel="Anonymous profile identifiers",
    )


def run_single_peaked_walsh_validator(num_obs, all_sp_ranks, plot_dir_root):
    validator = SPWalshValidator(len(all_sp_ranks[0]), all_outcomes=all_sp_ranks)
    run_validator(
        "Single-Peaked Walsh model",
        "sp_walsh",
        validator,
        num_obs,
        plot_dir_root,
        graph_xlabel="Single-peaked rank identifier",
    )


def run_single_peaked_conitzer_validator(num_obs, all_sp_ranks, plot_dir_root):
    validator = SPConitzerValidator(len(all_sp_ranks[0]), all_outcomes=all_sp_ranks)
    run_validator(
        "Single-Peaked Conitzer model",
        "sp_conitzer",
        validator,
        num_obs,
        plot_dir_root,
        graph_xlabel="Single-peaked rank identifier",
    )


def run_single_peaked_circle_validator(num_obs, all_sp_circle_ranks, plot_dir_root):
    validator = SPCircleValidator(
        len(all_sp_circle_ranks[0]), all_outcomes=all_sp_circle_ranks
    )
    run_validator(
        "Single-Peaked Circle model",
        "sp_circle",
        validator,
        num_obs,
        plot_dir_root,
        graph_xlabel="Circular single-peaked rank identifier",
    )


def run_ordinal_plackett_luce_validator(num_obs, alphas, all_ranks, plot_dir_root):
    if not isinstance(alphas[0], Iterable):
        alphas = [alphas]
    for i, a in enumerate(alphas):
        validator = PlackettLuceValidator(
            len(all_ranks[0]), alphas=a, all_outcomes=all_ranks
        )
        run_validator(
            "Plackett Luce model",
            f"plackett_luce_{i}",
            validator,
            num_obs,
            plot_dir_root,
        )


def run_single_crossing_validator(num_obs, all_sc_profiles, plot_dir_root):
    validator = SingleCrossingValidator(
        len(all_sc_profiles[0]),
        len(all_sc_profiles[0][0]),
        all_outcomes=all_sc_profiles,
    )
    run_validator(
        "Single-Crossing model",
        "single_crossing",
        validator,
        num_obs,
        plot_dir_root,
        graph_xlabel="Non-isomorphic single-crossing profile identifiers",
        graph_x_tick_labels=all_sc_profiles,
        graph_ordering="observed",
    )


def run_impartial_single_crossing_validator(num_obs, all_sc_profiles, plot_dir_root):
    validator = SingleCrossingImpartialValidator(
        len(all_sc_profiles[0]),
        len(all_sc_profiles[0][0]),
        all_outcomes=all_sc_profiles,
    )
    run_validator(
        "Impartial Single-Crossing model",
        "single_crossing_impartial",
        validator,
        num_obs,
        plot_dir_root,
        graph_xlabel="Non-isomorphic single-crossing profile identifiers",
    )


def run_ordinal_urn_validator(num_obs, alpha, all_anonymous_profiles, plot_dir_root):
    if not isinstance(alpha, Iterable):
        alpha = [alpha]
    for a in alpha:
        validator = UrnValidator(
            len(all_anonymous_profiles[0]),
            len(all_anonymous_profiles[0][0]),
            alpha=a,
            all_outcomes=all_anonymous_profiles,
        )
        run_validator(
            "Urn model",
            f"urn_{a}",
            validator,
            num_obs,
            plot_dir_root,
            graph_xlabel="Anonymous profile identifiers",
        )


def run_ordinal_euclidean_validator(num_obs, space_dimension, all_ranks, plot_dir_root):
    if not isinstance(space_dimension[0], Iterable):
        space_dimension = [space_dimension]
    for space, dimension in space_dimension:
        validator = OrdinalEuclideanValidator(
            len(all_ranks[0]), space, dimension, all_outcomes=all_ranks
        )
        run_validator(
            "Ordinal Euclidean model",
            f"ordinal_euclidean_{space}_{dimension}",
            validator,
            num_obs,
            plot_dir_root,
            graph_ordering="observed",
        )


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    num_observations = 1000000

    # -------------------
    # Ordinal Validators
    # -------------------

    ordinal_plot_dir_root = os.path.join("plots", "ordinal")
    os.makedirs(ordinal_plot_dir_root, exist_ok=True)

    all_ranks = get_all_ranks(num_candidates=5)
    all_sp_ranks = get_all_single_peaked_ranks(num_candidates=5)
    all_sp_circle_ranks = get_all_single_peaked_circle_ranks(num_candidates=5)
    all_anonymous_profiles = get_all_anonymous_profiles(num_voters=3, num_candidates=4)
    all_sc_profiles_non_iso = get_all_sc_profiles_non_iso(
        num_voters=3, num_candidates=4
    )

    # # Mallow's
    # all_phis = [0.1, 0.5, 0.8, 1]
    # run_ordinal_mallows_validator(
    #     num_observations, all_phis, all_ranks, ordinal_plot_dir_root
    # )
    #
    # # Impartial
    # run_ordinal_impartial_validator(num_observations, all_ranks, ordinal_plot_dir_root)
    # run_ordinal_impartial_anonymous_validator(
    #     num_observations, all_anonymous_profiles, ordinal_plot_dir_root
    # )
    # run_stratification_validator(num_observations, [0, 1/len(all_ranks[0]), 3/len(all_ranks[0]), 1], all_ranks, ordinal_plot_dir_root)
    #
    # # Plackett Luce
    # all_alphas = [
    #     [0.1] * len(all_ranks[0]),
    #     [1.0] + [0.1] * (len(all_ranks[0]) - 1),
    #     np.random.random(len(all_ranks[0])),
    # ]
    # run_ordinal_plackett_luce_validator(
    #     num_observations, all_alphas, all_ranks, ordinal_plot_dir_root
    # )
    #
    # Single-Peaked
    run_single_peaked_walsh_validator(
        num_observations, all_sp_ranks, ordinal_plot_dir_root
    )
    run_single_peaked_conitzer_validator(
        num_observations, all_sp_ranks, ordinal_plot_dir_root
    )
    run_single_peaked_circle_validator(
        num_observations, all_sp_circle_ranks, ordinal_plot_dir_root
    )
    #
    # # Single-Crossing
    # run_impartial_single_crossing_validator(
    #     num_observations, all_sc_profiles_non_iso, ordinal_plot_dir_root
    # )
    # run_single_crossing_validator(
    #     num_observations,
    #     get_all_sc_profiles_non_iso(num_voters=2, num_candidates=4),
    #     ordinal_plot_dir_root
    # )
    #
    # # Urn
    # run_ordinal_urn_validator(
    #     num_observations,
    #     [0, 1 / math.factorial(len(all_anonymous_profiles[0][0])), 0.5, 1],
    #     all_anonymous_profiles,
    #     ordinal_plot_dir_root,
    # )
    #
    # # Euclidean Ordinal
    # run_ordinal_euclidean_validator(
    #     num_observations,
    #     [
    #         (EuclideanSpace.UNIFORM, 2),
    #         (EuclideanSpace.SPHERE, 2),
    #         (EuclideanSpace.GAUSSIAN, 2),
    #     ],
    #     all_ranks,
    #     ordinal_plot_dir_root
    # )
