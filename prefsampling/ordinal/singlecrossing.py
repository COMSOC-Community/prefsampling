"""
Single-crossing preferences are such that the set of voters is one-dimensional: for a given
ordering of the voters and any pair of alternatives a, b, the two sets of voters corresponding to
the those preferring a over b and those preferring b over a form two segments of the ordering. In
other words, following the ordering of the voters, there is single crossing point such that all
voters to the left of this point prefer a over b and all voters to the right of this point prefer
b over a (or the symmetric case).
"""

from __future__ import annotations

from collections.abc import Iterable, Collection

import numpy as np

from prefsampling.combinatorics import (
    all_single_crossing_profiles,
    all_non_isomorphic_profiles,
    all_anonymous_profiles,
)
from prefsampling.inputvalidators import validate_num_voters_candidates, validate_int


@validate_num_voters_candidates
def single_crossing(
    num_voters: int, num_candidates: int, seed: int = None
) -> list[list[int]]:
    """
    Generates ordinal votes that are single-crossing. See `Elkind, Lackner, Peters (2022)
    <https://arxiv.org/abs/2205.09092>`_ for the definition.

    This sampler works as follows: a random single-crossing domain is generated, and votes are then
    randomly selected from this domain.

    The votes in the domain are generated one by one. The first vote is always
    `0 > 1 > 2 > ...`. For vote number `k`, all valid swaps of consecutive candidates
    are considered. A swap is valid if in vote number `k - 1` the two candidates had not
    already been swapped in any previous iterations. One valid swap is selected uniformly at random
    and vote number `k` correspond to vote number `k - 1` with the selected swap performed. After
    `m * (m-1) / 2 + 1` swaps, the final vote admit no valid swap (the final vote being
    `m > m - 1 > ...`). Importantly, the domain is ordered by number of swaps applied to the initial
    vote.

    To sample the votes, we consider the ordered domain. The first sampled vote is always
    `0 > 1 > 2 > ...`. Then, every successive vote is selected uniformly at random from all the
    votes further down in the domain than the previously selected vote (including the latter).

    This sampler only generates anonymous and neutral collections of votes. This means that
    the first vote is always `0 > 1 > 2 > ...` and that the votes are ordered lexicographically.
    To obtain a distribution over all single-crossing collections of votes, apply a random
    permutation of the candidate names and a random permutation of the position of the voters in the
    collection of votes.

    The sampler is inspired by the one used by `Szufa, Faliszewski, Skowron, Slinko, Talmon (2020)
    <https://www.ifaamas.org/Proceedings/aamas2020/pdfs/p1341.pdf>`_.

    Note that the probability distribution over single-crossing profiles yielded by this procedure
    is unknown. For a uniform distribution over outcomes, see the
    :py:func:`~prefsampling.ordinal.single_crossing_impartial` function. The uniform sampler can
    become very slow as the number of candidates increases. The difficulty lies in computing the
    probability distribution with which to select votes further down in the domain (instead of a
    uniform distribution as we use here).

    Note that for a given number of voters, votes are not sampled independently.

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        seed : int, default: :code:`None`
            Seed for numpy random number generator.

    Returns
    -------
        list[list[int]]
            Ordinal votes.

    Examples
    --------

        .. testcode::

            from prefsampling.ordinal import single_crossing

            # Sample a single-crossing profile with 2 voters and 3 candidates.
            single_crossing(2, 3)

            # For reproducibility, you can set the seed.
            single_crossing(2, 3, seed=1002)

    Validation
    ----------
        The theoretical distribution of this sampler is unknown. We just plot the observed
        frequencies.

        .. image:: ../validation_plots/ordinal/single_crossing_None.png
            :width: 800
            :alt: Observed versus theoretical frequencies for impartial single-crossing model


    References
    ----------
        `An exploration in the theory of optimal income taxation
        <https://www.jstor.org/stable/2296779>`_,
        *James Mirrlees*,
        Review of Economic Studies, 38:175–208, 1971.

        `Voting over income tax schedules
        <https://www.sciencedirect.com/science/article/pii/0047272777900056>`_,
        *Kevin W.S. Roberts*,
        Journal of Public Economics, 8(3):329–340, 1977.

        `Drawing a map of elections in the space of statistical cultures
        <https://ifaamas.org/Proceedings/aamas2020/pdfs/p1341.pdf>`_,
        *Stanisław Szufa, Piotr Faliszewski, Piotr Skowron, Arkadii Slinko and Nimrod Talmon*,
        Proceedings of AAMAS-2020, pages 1341–1349, 2020.
    """
    rng = np.random.default_rng(seed)

    domain_size = int(num_candidates * (num_candidates - 1) / 2 + 1)
    domain = [list(range(num_candidates))]
    for line in range(1, domain_size):
        all_swap_indices = [
            (j, j + 1)
            for j in range(num_candidates - 1)
            if domain[line - 1][j] < domain[line - 1][j + 1]
        ]
        swap_indices = all_swap_indices[rng.integers(0, len(all_swap_indices))]

        new_line = domain[line - 1].copy()
        new_line[swap_indices[0]] = domain[line - 1][swap_indices[1]]
        new_line[swap_indices[1]] = domain[line - 1][swap_indices[0]]
        domain.append(new_line)

    votes = []
    last_sampled_index = 0
    votes.append(domain[0])
    for i in range(1, num_voters):
        index = rng.integers(last_sampled_index, domain_size)
        votes.append(domain[index])
        last_sampled_index = index

    return votes


class SingleCrossingNode:
    """
    Node in the graph of all the single-crossing set of votes. A node represents a vote and is
    connected to all the votes that can be accessed by valid swaps of the candidates. The
    resulting graph is a directed acyclic graph.
    """

    def __init__(self, vote, seed=None):
        self.vote = vote
        self.vote_as_tuple = tuple(vote)  # used for hashing and comparison
        self.next = []  # all next nodes reachable via single swap
        self.all_next = set()  # all next nodes reachable via any number of swaps
        self.election_count = {}  # memoization of the count
        self.rng = np.random.default_rng(seed)  # the random number generator used

    def generate_all_next(self):
        """
        Unravels the graph and populate the set of nodes that are accessible from the current node.
        """
        if self.all_next:
            return
        for node in self.next:
            node.generate_all_next()
            self.all_next |= node.all_next
        self.all_next.add(self)

    def count_elections(self, n):
        """
        Counts the number of elections. This number is used to define the probability distribution
        over the nodes when sampling votes.
        """
        if n < 0:
            raise ValueError("The number of agents should be more at least 0.")
        if n < 2:
            return n
        count = self.election_count.get(n, None)
        if count is None:
            count = sum(node.count_elections(n - 1) for node in self.all_next)
            self.election_count[n] = count
        return count

    def sample_votes(self, n):
        """
        Samples a collection of votes by selecting with the correct probability distribution the
        next vote to select based on the current node (moving down the graph).
        """
        election = [self.vote]
        if n == 1:
            return election
        nodes_distribution = []
        next_nodes = []
        for node in self.all_next:
            next_nodes.append(node)
            nodes_distribution.append(node.count_elections(n - 1))
        nodes_distribution = np.array(nodes_distribution, dtype=np.float64)
        nodes_distribution /= nodes_distribution.sum()
        random_node = self.rng.choice(next_nodes, p=nodes_distribution)
        return election + random_node.sample_votes(n - 1)

    def __hash__(self):
        return hash(self.vote_as_tuple)

    def __eq__(self, other):
        return self.vote_as_tuple == other

    def __repr__(self):
        return f"Node{self.vote_as_tuple}"


@validate_num_voters_candidates
def single_crossing_impartial(num_voters, num_candidates, seed=None) -> list[list[int]]:
    """
    Generates ordinal votes that satisfy the single-crossing property. See `Elkind, Lackner,
    Peters (2022) <https://arxiv.org/abs/2205.09092>`_ for the definition. This sampler ensures
    that all single-crossing profiles are equally likely to be generated. Note that its running time
    is exponential in the number of candidates.

    This sampler initially generates all single-crossing domains. The collection of vote is then
    sampled from a domain. Since all domains have been generated, via some counting we can ensure
    that all profile are equally likely to be generated.

    This sampler only generates non-isomorphic and anonymous collection of votes. This means that
    the first vote is always `0 > 1 > 2 > ...` and that the votes are ordered lexicographically.
    To obtain a distribution over all single-crossing collections of votes, apply a random
    permutation of the candidate names and a random permutation of the position of the voters in the
    collection of votes.

    This sampler can be very slow for large number of candidates. For efficient sampling---but
    without theoretical guarantees on the distribution of outcomes---consider using
    :py:func:`~prefsampling.ordinal.singlecrossing.single_crossing`.

    Note that for a given number of voters, votes are not sampled independently.

    This sampler was developed by `Piotr Faliszewski <https://home.agh.edu.pl/~faliszew/>`_.

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        seed : int, default: :code:`None`
            Seed for numpy random number generator.

    Returns
    -------
        list[list[int]]
            Ordinal votes.

    Examples
    --------

        .. testcode::

            from prefsampling.ordinal import single_crossing_impartial

            # Sample a single-crossing profile with 2 voters and 3 candidates.
            single_crossing_impartial(2, 3)

            # For reproducibility, you can set the seed.
            single_crossing_impartial(2, 3, seed=1002)

        .. code-block::

            # This already takes a very long time... (only 10 candidates)
            single_crossing_impartial(2, 10)

    Validation
    ----------
        This sampler is meant to follow a uniform probability distribution over all non-isomorphic
        and anonymous single-crossing profiles.

        .. image:: ../validation_plots/ordinal/single_crossing_impartial_None.png
            :width: 800
            :alt: Observed versus theoretical frequencies for impartial single-crossing model

    References
    ----------
        `An exploration in the theory of optimal income taxation
        <https://www.jstor.org/stable/2296779>`_,
        *James Mirrlees*,
        Review of Economic Studies, 38:175–208, 1971.

        `Voting over income tax schedules
        <https://www.sciencedirect.com/science/article/pii/0047272777900056>`_,
        *Kevin W.S. Roberts*,
        Journal of Public Economics, 8(3):329–340, 1977.
    """
    top_vote = list(range(num_candidates))
    top_node = SingleCrossingNode(top_vote, seed=seed)
    vote_node_map = {top_node.vote_as_tuple: top_node}

    def graph_builder(node):
        vote = list(node.vote)
        for j in range(num_candidates - 1):
            if vote[j] < vote[j + 1]:
                new_vote = vote.copy()
                new_vote[j], new_vote[j + 1] = new_vote[j + 1], new_vote[j]
                vote_as_tuple = tuple(new_vote)
                if not (vote_as_tuple in vote_node_map):
                    new_node = SingleCrossingNode(new_vote, seed=seed)
                    vote_node_map[vote_as_tuple] = new_node
                    graph_builder(new_node)
                new_node = vote_node_map[vote_as_tuple]
                node.next.append(new_node)

    graph_builder(top_node)

    top_node.generate_all_next()
    top_node.count_elections(num_voters)

    votes = top_node.sample_votes(num_voters)
    return votes


def impartial_theoretical_distribution(
    num_voters: int = None,
    num_candidates: int = None,
    sc_profiles: Iterable[Collection[tuple[int]]] = None,
) -> dict:
    if sc_profiles is None:
        if num_candidates is None:
            raise ValueError(
                "If you do not provide the collection of single-crossing profiles, "
                "you need to provide the number of candidates."
            )
        if num_voters is None:
            raise ValueError(
                "If you do not provide the collection of single-crossing profiles, "
                "you need to provide the number of voters."
            )
        validate_int(num_candidates, lower_bound=1)
        validate_int(num_voters, lower_bound=1)
        sc_profiles = all_single_crossing_profiles(
            num_voters,
            num_candidates,
            profiles=all_non_isomorphic_profiles(
                num_voters,
                num_candidates,
                profiles=all_anonymous_profiles(
                    num_voters,
                    num_candidates,
                ),
            ),
            fix_order=True,
        )
    return {o: 1 / len(sc_profiles) for o in sc_profiles}
