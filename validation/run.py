import logging
import os

from validation.approval.impartial import (
    ApprovalImpartialValidator,
    ApprovalImpartialConstantSizeValidator,
)
from validation.approval.noise import ApprovalNoiseValidator
from validation.approval.resampling import (
    ApprovalResamplingValidator,
    ApprovalDisjointResamplingValidator,
    ApprovalMovingResamplingValidator,
)

from prefsampling.ordinal import TreeSampler
from prefsampling.tree.schroeder import (
    all_schroeder_tree,
    schroeder_tree_lescanne,
    schroeder_tree,
    schroeder_tree_brute_force,
)
from validation.ordinal.didi import DidiValidator
from validation.ordinal.euclidean import (
    OrdinalEuclideanValidator,
    OrdinalEuclideanValidatorUniform,
    OrdinalEuclideanValidatorLine,
)
from validation.ordinal.groupseparable import GroupSeparableValidator
from validation.ordinal.impartial import (
    OrdinalImpartialValidator,
    OrdinalImpartialAnonymousValidator,
    OrdinalStratificationValidator,
    OrdinalStratificationUniformValidator,
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
from validation.ordinal.urn import OrdinalUrnValidator
from validation.point.ball_uniform import plot_ball_uniform
from validation.point.ball_resampling import plot_ball_resampling
from validation.point.cube_uniform import plot_cube_uniform
from validation.point.gaussian import plot_gaussian
from validation.point.sphere_uniform import plot_sphere_uniform
from validation.tree.schroeder import (
    SchroederValidator,
    SchroederLescanneValidator,
    SchroederBruteForceValidator,
)

ALL_APPROVAL_VALIDATORS = [
    # ApprovalImpartialValidator(),
    # ApprovalImpartialConstantSizeValidator(),
    # ApprovalNoiseValidator(),
    # ApprovalResamplingValidator(),
    # ApprovalDisjointResamplingValidator(),
    # ApprovalMovingResamplingValidator()
]

ALL_ORDINAL_VALIDATORS = [
    # OrdinalImpartialValidator(),
    # OrdinalImpartialAnonymousValidator(),
    # OrdinalStratificationValidator(),
    # OrdinalStratificationUniformValidator(),
    # OrdinalMallowsValidator(),
    # PlackettLuceValidator(),
    # DidiValidator(),
    # OrdinalEuclideanValidator(),
    # OrdinalEuclideanValidatorUniform(),
    # OrdinalEuclideanValidatorLine(),
    # SPWalshValidator(),
    # SPConitzerValidator(),
    # SPCircleValidator(),
    # SingleCrossingValidator(),
    # SingleCrossingImpartialValidator(),
    # GroupSeparableValidator(),
    # OrdinalUrnValidator(),
]

ALL_TREE_VALIDATORS = [
    # SchroederValidator(),
    # SchroederLescanneValidator(),
    # SchroederBruteForceValidator(),
]

if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    num_observations = 1000000

    for validator_list, nickname in [
        (ALL_APPROVAL_VALIDATORS, "approval"),
        (ALL_ORDINAL_VALIDATORS, "ordinal"),
        (ALL_TREE_VALIDATORS, "tree"),
    ]:
        csv_dir_root = os.path.join("csv", nickname)
        os.makedirs(csv_dir_root, exist_ok=True)
        plot_dir_root = os.path.join("plots", nickname)
        os.makedirs(plot_dir_root, exist_ok=True)

        # for validator in validator_list:
        #     validator.write_csv(num_observations, csv_dir_root)
        for validator in validator_list:
            validator.plot_frequencies(
                csv_dir_root, plot_dir_root, ordering="theoretical-observed"
            )

    # plot_dir_root = os.path.join("plots", "point")
    # os.makedirs(plot_dir_root, exist_ok=True)
    # plot_ball_uniform(os.path.join(plot_dir_root, "ball-uniform"))
    # plot_sphere_uniform(os.path.join(plot_dir_root, "sphere-uniform"))
    # plot_ball_resampling(os.path.join(plot_dir_root, "ball-resampling"))
    # plot_cube_uniform(os.path.join(plot_dir_root, "cube"))
    # plot_gaussian(os.path.join(plot_dir_root, "gaussian"))
