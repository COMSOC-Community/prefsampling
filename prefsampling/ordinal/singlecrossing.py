import numpy as np

from prefsampling.decorators import validate_num_voters_candidates


@validate_num_voters_candidates
def single_crossing(
        num_voters: int, num_candidates: int, seed: int = None
) -> np.ndarray:
    """
    Generates ordinal votes that are single-crossing. See `Elkind, Lackner, Peters (2022)
    <https://arxiv.org/abs/2205.09092>`_ for the definition.

    This sampler works as follows. We generate a random domain of single-crossing, and then randomly
    selects the votes from the domain. The votes in the domain are generated one by one. The first
    vote is always `0 > 1 > 2 > ...`. For vote number `k`, all valid swaps of consecutive candidates
    are considered where a swap is only valid if in vote number `k - 1` the two candidates had not
    already been swapped in previous iterations. One valid swap is selected uniformly at random
    and performed on vote number `k`. One can check that after `m * (m-1) / 2 + 1` votes have been
    generated, no valid swap exists (the final vote being `m > m - 1 > ...`). Once the domain is
    set, we select for each voter one vote from the domain uniformly at random.

    TODO: This is wrong

    This procedure ensures that every set of single-crossing votes for `num_voters` and
    `num_candidates` is equally likely to occur.

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        seed : int
            Seed for numpy random number generator.

    Returns
    -------
        np.ndarray
            Ordinal votes.
    """
    rng = np.random.default_rng(seed)

    domain_size = int(num_candidates * (num_candidates - 1) / 2 + 1)
    domain = [np.arange(num_candidates)]

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

    votes = np.zeros([num_voters, num_candidates], dtype=int)
    vote_indices = np.sort(rng.choice(np.arange(domain_size), size=num_voters))
    perm = list(domain[vote_indices[0]])
    for i, index in enumerate(vote_indices):
        votes[i, :] = [perm.index(c) for c in domain[index]]
    return votes


class SingleCrossingNode:
    """
    Node in the graph of all the single-crossing set of votes.
    """

    def __init__(self, vote, seed=None):
        self.vote = tuple(vote)
        self.next = []  # all next nodes reachable via single swap
        self.all_next = set()  # all next nodes reachable via any number of swaps
        self.domains = -1  # number of different domains that can be
        self.election_count = {}
        self.rng = np.random.default_rng(seed)

    def generate_all_next(self):
        if self.all_next:
            return
        for node in self.next:
            node.generate_all_next()
            self.all_next |= node.all_next
        self.all_next.add(self.vote)

    def count_elections(self, n, nodes):
        if n == 0:
            return 0
        if n == 1:
            return 1
        count = self.election_count.get(n, None)
        if count is None:
            count = sum(nodes[vote].count_elections(n - 1, nodes) for vote in self.all_next)
            self.election_count[n] = count
        return count

    def sample_election(self, n, nodes):
        election = [self.vote]
        if n == 1:
            return election
        nodes_distribution = []
        next_nodes = []
        for vote in self.all_next:
            node = nodes[vote]
            next_nodes.append(node)
            nodes_distribution.append(node.count_elections(n - 1, nodes))
        nodes_distribution = np.array(nodes_distribution, dtype=np.float64)
        nodes_distribution /= nodes_distribution.sum()
        random_node = self.rng.choice(next_nodes, p=nodes_distribution)
        return election + random_node.sample_election(n - 1, nodes)


def single_crossing_uniform(num_voters, num_candidates):
    nodes = {}
    top_vote = np.arange(num_candidates)
    top_node = SingleCrossingNode(top_vote)
    nodes[tuple(top_vote)] = top_node

    def graph_builder(node):
        vote = np.array(node.vote)
        for i in range(num_candidates - 1):
            if vote[i] < vote[i + 1]:
                new_vote = vote.copy()
                new_vote[i], new_vote[i + 1] = new_vote[i + 1], new_vote[i]
                tuple_vote = tuple(new_vote)
                if not (tuple_vote in nodes):
                    new_node = SingleCrossingNode(new_vote)
                    nodes[tuple_vote] = new_node
                    graph_builder(new_node)
                new_node = nodes[tuple_vote]
                node.next.append(new_node)

    graph_builder(top_node)

    top_node.generate_all_next()
    top_node.count_elections(num_voters, nodes)

    votes = top_node.sample_election(num_voters, nodes)
    return votes
