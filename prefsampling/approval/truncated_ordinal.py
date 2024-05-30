from __future__ import annotations

from collections.abc import Callable, Iterable

from prefsampling.inputvalidators import validate_num_voters_candidates


@validate_num_voters_candidates
def truncated_ordinal(
    num_voters: int,
    num_candidates: int,
    rel_num_approvals: float | Iterable[float],
    ordinal_sampler: Callable,
    ordinal_sampler_parameters: dict,
    seed: int = None,
) -> list[set[int]]:
    """
    Generates approval votes by truncating ordinal votes sampled from a given ordinal sampler.

    The process is as follows: ordinal votes are sampled from the ordinal sampler. These votes are
    then truncated in a way that each approval vote consists in the
    `rel_num_approvals * num_candidates` first candidates of the ordinal vote.

    Parameters
    ----------
        num_voters: int
            Number of voters
        num_candidates: int
            Number of candidates
        rel_num_approvals: float | Iterable[float],
            Ratio of approved candidates. If an iterable is provided, then there must one such value
            per voter.
        ordinal_sampler: Callable
            The ordinal sampler to be used.
        ordinal_sampler_parameters: dict
            The arguments passed ot the ordinal sampler. The num_voters, num_candidates and seed
            parameters are overridden by those passed to this function.
        seed : int, default: :code:`None`
            Seed for numpy random number generator.

    Returns
    -------
        list[set[int]]
            Approval votes

    Examples
    --------

        .. testcode::

            from prefsampling.approval import truncated_ordinal
            from prefsampling.ordinal import mallows

            # Sample votes from Mallows' model and convert them to approval ballots, there are
            # 4 voters and 5 candidates, rel_num_approvals is 0.5
            truncated_ordinal(4, 5, 0.5, mallows, {"phi": 0.4})

            # You can also specify one rel_num_approvals per voter
            truncated_ordinal(4, 5, [0.3, 0.2, 0.4, 0.7], mallows, {"phi": 0.4})

            # For reproducibility, you can set the seed which is then passed to the ordinal sampler.
            truncated_ordinal(4, 5, 0.5, mallows, {"phi": 0.4}, seed=756)

            # If you pass num_voters, num_candidates or seed to the ordinal sampler,
            # they are erased
            votes = truncated_ordinal(
                4,
                5,
                1,  # All candidates will be approved
                mallows,
                {"num_voters": 10, "num_candidates": 15, "phi": 0.4}
            )
            assert len(votes) == 4  # And not 10
            assert len(votes[0]) == 5  # And not 15

            # Parameter rel_num_approvals needs to be in [0, 1]
            try:
                truncated_ordinal(4, 5, 1.5, mallows, {"phi": 0.4})
            except ValueError:
                pass
            try:
                truncated_ordinal(4, 5, -0.5, mallows, {"phi": 0.4})
            except ValueError:
                pass

    References
    ----------

        `How to Sample Approval Elections?
        <https://www.ijcai.org/proceedings/2022/71>`_,
        *Stanisław Szufa, Piotr Faliszewski, Łukasz Janeczko, Martin Lackner, Arkadii Slinko,
        Krzysztof Sornat and Nimrod Talmon*,
        Proceedings of the International Joint Conference on Artificial Intelligence, 2022.
    """
    if isinstance(rel_num_approvals, Iterable):
        unique_vote_length = False
        if min(rel_num_approvals) < 0 or max(rel_num_approvals) > 1:
            raise ValueError(
                "Incorrect value of rel_num_approvals. All values should be in [0, 1]"
            )
        vote_length = [int(r * num_candidates) for r in rel_num_approvals]
        if len(vote_length) != num_voters:
            raise ValueError(
                "If you provide an iterable as rel_num_approvals, there they should "
                "be exactly one value per voter, no more, no less."
            )
    else:
        unique_vote_length = True
        if rel_num_approvals < 0 or 1 < rel_num_approvals:
            raise ValueError(
                f"Incorrect value of rel_num_approvals: {rel_num_approvals}. Value should"
                f" be in [0, 1]"
            )
        vote_length = int(rel_num_approvals * num_candidates)

    ordinal_sampler_parameters["num_voters"] = num_voters
    ordinal_sampler_parameters["num_candidates"] = num_candidates
    ordinal_sampler_parameters["seed"] = seed
    ordinal_votes = ordinal_sampler(**ordinal_sampler_parameters)

    votes = []
    for i, vote in enumerate(ordinal_votes):
        if unique_vote_length:
            votes.append({int(c) for c in vote[0:vote_length]})
        else:
            votes.append({int(c) for c in vote[0 : vote_length[i]]})
    return votes
