"""
Impartial cultures are statistical cultures in which all outcomes are equally likely to be
generated.
"""

from __future__ import annotations

from collections.abc import Sequence, Iterable

import numpy as np

from prefsampling.inputvalidators import validate_num_voters_candidates


@validate_num_voters_candidates
def impartial(
    num_voters: int, num_candidates: int, p: float | Iterable[float], seed: int = None
) -> list[set]:
    """
    Generates approval votes from impartial culture.

    Under the approval culture, when generating a single vote, each candidate has the same
    probability :code:`p` of being approved. This models ensures that the average number of
    approved candidate per voter is `p * num_candidate`.

    A collection of `num_voters` vote is generated independently and identically following the
    process described above.

    See :py:func:`~prefsampling.approval.impartial.impartial_constant_size` for a version in which
    all voters approve of the same number of candidates.

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        p : float | Iterable[float]
            Probability of approving of any given candidates. If a sequence is passed, there is one
            such probability per voter.
        seed : int, default: :code:`None`
            Seed for numpy random number generator.

    Returns
    -------
        list[set]
            Approval votes.

    Examples
    --------

        **Use a global** :code:`p` **for all voters**

        .. testcode::

            from prefsampling.approval import impartial

            # Sample from an impartial culture with 2 voters and 3 candidates where
            # a candidate is approved with 60% probability.
            impartial(2, 3, 0.6)

            # For reproducibility, you can set the seed.
            impartial(2, 3, 0.6, seed=1002)

            # Parameter p needs to be in [0, 1]
            try:
                impartial(2, 3, 1.6)
            except ValueError:
                pass
            try:
                impartial(2, 3, -0.6)
            except ValueError:
                pass

        **Use an individual** :code:`p` **per voter**

        .. testcode::

            from prefsampling.approval import impartial

            # Sample from an impartial culture with 2 voters and 3 candidates with
            # p=0.6 for the first voter and p=0.2 for the second.
            impartial(2, 3, [0.6, 0.2])

            # For reproducibility, you can set the seed.
            impartial(2, 3, [0.6, 0.2], seed=1002)

            # There need to be one p per voter (and no more)
            try:
                impartial(2, 3, [0.6, 0.2, 0.9])
            except ValueError:
                pass
            try:
                impartial(2, 3, [0.6])
            except ValueError:
                pass

            # All individual p's need to be in [0, 1]
            try:
                impartial(2, 3, [0.6, -0.2])
            except ValueError:
                pass
            try:
                impartial(2, 3, [1.6, 0.2])
            except ValueError:
                pass

    Validation
    ----------

        We only validate the model with a single voter thus the distinction between individual and
        global does not matter here. Call :math:`p` the probability of approving any candidate,
        then the probability of generating a given approval ballot of size :math:`k` is
        :math:`p^k \\times (1 - p)^{m - k}`, where :math:`m` is the number of candidates.

        .. image:: ../validation_plots/approval/impartial_0_3.png
            :width: 800
            :alt: Observed versus theoretical frequencies for an impartial culture 0.3

        .. image:: ../validation_plots/approval/impartial_0_5.png
            :width: 800
            :alt: Observed versus theoretical frequencies for an impartial culture 0.5

        .. image:: ../validation_plots/approval/impartial_0_7.png
            :width: 800
            :alt: Observed versus theoretical frequencies for an impartial culture 0.7

    References
    ----------

        `An Experimental View on Committees Providing Justified Representation
        <https://www.ijcai.org/proceedings/2019/16>`_,
        *Robert Bredereck, Piotr Faliszewski, Andrzej Kaczmarczyk and Rolf Niedermeier*,
        Proceedings of the International Joint Conference on Artificial Intelligence, 2019.

        `How to Sample Approval Elections?
        <https://www.ijcai.org/proceedings/2022/71>`_,
        *Stanisław Szufa, Piotr Faliszewski, Łukasz Janeczko, Martin Lackner, Arkadii Slinko,
        Krzysztof Sornat and Nimrod Talmon*,
        Proceedings of the International Joint Conference on Artificial Intelligence, 2022.

        `Price of Fairness in Budget Division and Probabilistic Social Choice
        <https://ojs.aaai.org/index.php/AAAI/article/view/5594>`_,
        * Marcin Michorzewski, Dominik Peters and Piotr Skowron*,
        Proceedings of the AAAI Conference on Artificial Intelligence, 2020.
    """
    unique_p = True
    if isinstance(p, Iterable):
        p = tuple(p)
        if len(p) != num_voters:
            raise ValueError(
                "In the impartial model, if parameter p is a sequence, it needs to"
                "have as many elements as there are voters"
            )
        for prob in p:
            if prob < 0 or 1 < prob:
                raise ValueError(
                    f"Incorrect value of p: {prob}. All value of the sequence "
                    f"should be in [0, 1]"
                )
        unique_p = False
    if unique_p and (p < 0 or 1 < p):
        raise ValueError(f"Incorrect value of p: {p}. Value should be in [0, 1]")

    rng = np.random.default_rng(seed)

    votes = [
        set(
            j
            for j in range(num_candidates)
            if rng.random() <= (p if unique_p else p[i])
        )
        for i in range(num_voters)
    ]

    return votes


@validate_num_voters_candidates
def impartial_constant_size(
    num_voters: int,
    num_candidates: int,
    rel_num_approvals: float | Iterable[float],
    seed: int = None,
) -> list[set]:
    """
    Generates approval votes from impartial culture with constant size.

    Under this culture, all ballots are of size `⌊rel_num_approvals * num_candidates⌋`. The ballot
    is selected uniformly at random over all ballots of size `⌊rel_num_approvals * num_candidates⌋`.

    A collection of `num_voters` vote is generated independently and identically following the
    process described above.

    See :py:func:`~prefsampling.approval.impartial.impartial` for a version in the probability of
    approving any candidate is constant and independent.

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        rel_num_approvals : float | Iterable[float]
            Proportion of approved candidates in a ballot. If a sequence is passed, there is one
            such proportion per voter.
        seed : int, default: :code:`None`
            Seed for numpy random number generator.

    Returns
    -------
        list[set]
            Approval votes.

    Examples
    --------

        **Use a global** :code:`rel_num_approvals` **for all voters**

        .. testcode::

            from prefsampling.approval import impartial_constant_size

            # Sample from an impartial culture with 2 voters and 3 candidates where
            # all voters approve of 60% of the candidates (in this case 1).
            impartial_constant_size(2, 3, 0.6)

            # For reproducibility, you can set the seed.
            impartial_constant_size(2, 3, 0.6, seed=1002)

            # Parameter rel_num_approvals needs to be in [0, 1]
            try:
                impartial_constant_size(2, 3, 1.6)
            except ValueError:
                pass
            try:
                impartial_constant_size(2, 3, -0.6)
            except ValueError:
                pass

        **Use an individual** :code:`rel_num_approvals` **per voter**

        .. testcode::

            from prefsampling.approval import impartial_constant_size

            # Sample from a constant size impartial culture with 2 voters and 3 candidates with
            # p=0.6 for the first voter and p=0.2 for the second.
            impartial_constant_size(2, 3, [0.6, 0.2])

            # For reproducibility, you can set the seed.
            impartial_constant_size(2, 3, [0.6, 0.2], seed=1002)

            # There need to be one rel_num_approvals per voter (and no more)
            try:
                impartial_constant_size(2, 3, [0.6, 0.2, 0.9])
            except ValueError:
                pass
            try:
                impartial_constant_size(2, 3, [0.6])
            except ValueError:
                pass

            # All individual rel_num_approvals' need to be in [0, 1]
            try:
                impartial_constant_size(2, 3, [0.6, -0.2])
            except ValueError:
                pass
            try:
                impartial_constant_size(2, 3, [1.6, 0.2])
            except ValueError:
                pass

    Validation
    ----------

        We only validate the model with a single voter thus the distinction between individual and
        global does not matter here. For a given value of :code:`rel_num_approvals`, let
        :math:`s = \\lfloor \\text{rel\\_num\\_approvals} \\times m \\rfloor` be the size of the
        approval ballots. Then, the probability of generating a given approval ballot of size
        :math:`k` is 0 if :math:`k \\neq s`, and :math:`\\frac{1}{\\binom{m}{s}}` otherwise,
        where :math:`m` is the number of candidates.

        .. image:: ../validation_plots/approval/impartial_constant_size_0_3.png
            :width: 800
            :alt: Observed versus theoretical frequencies for an impartial culture constant 0.3

        .. image:: ../validation_plots/approval/impartial_constant_size_0_5.png
            :width: 800
            :alt: Observed versus theoretical frequencies for an impartial culture constant 0.5

        .. image:: ../validation_plots/approval/impartial_constant_size_0_7.png
            :width: 800
            :alt: Observed versus theoretical frequencies for an impartial culture constant 0.7

    References
    ----------

        `A Quantitative Analysis of Multi-Winner Rules
        <https://www.ijcai.org/proceedings/2019/58>`_,
        *Martin Lackner and Piotr Skowron*,
        Proceedings of the International Joint Conference on Artificial Intelligence, 2019.

    """

    unique_rel_num_approvals = True
    if isinstance(rel_num_approvals, Iterable):
        num_approvals = []
        for prop in rel_num_approvals:
            num_approvals.append(int(prop * num_candidates))
            if prop < 0 or 1 < prop:
                raise ValueError(
                    f"Incorrect value of rel_num_approvals: {prop}. All value of the "
                    "sequence should be in [0, 1]"
                )
        if len(num_approvals) != num_voters:
            raise ValueError(
                "In the impartial model with constant size, if parameter "
                "rel_num_approvals is an iterable, it needs to have as many elements "
                "as there are voters."
            )
        unique_rel_num_approvals = False
    else:
        if rel_num_approvals < 0 or 1 < rel_num_approvals:
            raise ValueError(
                f"Incorrect value of rel_num_approvals: {rel_num_approvals}. Value should"
                f" be in [0,1]"
            )
        num_approvals = int(rel_num_approvals * num_candidates)

    rng = np.random.default_rng(seed)
    candidate_range = range(num_candidates)
    votes = [
        set(
            rng.choice(
                candidate_range,
                size=num_approvals if unique_rel_num_approvals else num_approvals[i],
                replace=False,
            )
        )
        for i in range(num_voters)
    ]

    return votes
