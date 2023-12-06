import math
import queue
from enum import Enum
from itertools import chain

import numpy as np

from prefsampling.decorators import validate_num_voters_candidates


class DecompositionTree(Enum):
    """
    Constants use to represent different types of decomposition trees for group separable
    preferences.
    """

    RANDOM = 1
    """
    Random trees
    """

    CATERPILLAR = 2
    """
    Caterpillar trees
    """

    BALANCED = 3
    """
    Balanced trees
    """


@validate_num_voters_candidates
def group_separable(
    num_voters: int,
        num_candidates: int,
        tree: DecompositionTree = DecompositionTree.RANDOM,
        seed: int = None
):
    """
    Samplers for group separable votes. For the definition of group-seprable preferences, see
    `Elkind, Lackner, Peters (2022) <https://arxiv.org/abs/2205.09092>`_.

    This sampler yields a uniform distribution over group separable collections of votes. It
    implements the algorithm presented by `Faliszewski, Karpov, Obraztsova (2022)
    <https://link.springer.com/article/10.1007/s10458-022-09549-7>`_

    Note that for a given number of voters, votes are not sampled independently.

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        tree : DecompositionTree, default: :py:class:`~prefsampling.ordinal.groupseparable.DecompositionTree.RANDOM`
            Type of decomposition tree considered. Should be a constant defined in the
            :py:class:`~prefsampling.ordinal.groupseparable.DecompositionTree` enumeration.
        seed : int, default: :code:`None`
            Seed for numpy random number generator.

    Returns
    -------
        np.ndarray
            Ordinal votes.
    """

    rng = np.random.default_rng(seed)

    if tree == DecompositionTree.RANDOM:
        buckets = np.zeros(num_candidates - 1)
        for r in range(1, num_candidates):
            buckets[r - 1] = _number_decomposition_tree(num_candidates, r)
        buckets /= buckets.sum()
        num_internal_nodes = rng.choice(len(buckets), p=buckets) + 1

        decomposition_tree = _decompose_tree(
            num_candidates, num_internal_nodes, rng
        )
    elif tree == DecompositionTree.CATERPILLAR:
        decomposition_tree = _generate_caterpillar(num_candidates)

    elif tree == DecompositionTree.BALANCED:
        decomposition_tree = _generate_balanced_tree(num_candidates)
    else:
        raise ValueError(
            "The `tree` argument needs to be one of the constant defined in the "
            "ordinal.DecompositionTree enumeration. Choices are: "
            + ", ".join(str(s) for s in DecompositionTree)
        )

    all_inner_nodes = _get_all_inner_nodes(decomposition_tree)

    votes = np.zeros((num_voters, num_candidates), dtype=int)
    for i in range(num_voters):
        signature = [rng.choice([0, 1]) for _ in range(len(all_inner_nodes))]

        for j, node in enumerate(all_inner_nodes):
            node.reverse = signature[j]

        raw_vote = _sample_a_vote(decomposition_tree)
        votes[i] = np.array(tuple(int(candidate.replace("x", "")) for candidate in raw_vote))

        for j, node in enumerate(all_inner_nodes):
            node.reverse = False

    return votes


def _number_decomposition_tree(m: int, r: int) -> float:
    """
    Returns the number of decomposition trees given the number of candidates `m` and the number
    of internal nodes `r` based on the formula from `Karpov (2019)
    <https://link.springer.com/article/10.1007/s10726-019-09621-w>`_
    """
    return math.comb(m - 1, r) * math.comb(m - 1 + r, m) / (m - 1)


def _decompose_tree(num_leaves, num_internal_nodes, rng):
    """
    Generates a random decomposition tree with suitable number of leaves and internal nodes.
    The algorithm was presented by `Alonso, Rémy, Schott (1997)
    <https://www.sciencedirect.com/science/article/pii/S0020019097001749>`_ building on `Alonso,
    Rémy, Schott (1997) <https://link.springer.com/article/10.1007/BF02522824>`_

    """

    num_nodes = num_leaves + num_internal_nodes
    seq, _ = _generate_tree(num_nodes, num_internal_nodes, rng)

    seq = _cycle_lemma(seq)

    tree = _turn_pattern_into_tree(seq)

    return tree


def _get_all_inner_nodes(node):
    if node.leaf:
        return []
    output = [[node]]
    for i in range(len(node.children)):
        output.append(_get_all_inner_nodes(node.children[i]))
    return list(chain.from_iterable(output))


def _sample_a_vote(node, reverse=False):
    """
    Recursively sample a single vote.
    """
    if node.leaf:
        return [node.election_id]
    output = []
    if reverse == node.reverse:
        for i in range(len(node.children)):
            output.append(_sample_a_vote(node.children[i]))
    else:
        for i in range(len(node.children)):
            output.append(
                _sample_a_vote(node.children[len(node.children) - 1 - i], reverse=True)
            )

    return list(chain.from_iterable(output))


class GroupSeparableNode:
    total_num_leaf_descendants = 0

    def __init__(self, election_id):
        self.election_id = election_id
        self.parent = None
        self.children = []
        self.leaf = True
        self.reverse = False

        self.left = 0
        self.right = 0

        self.num_leaf_descendants = None
        self.depth = None
        self.scheme = {}
        self.scheme_1 = {}
        self.scheme_2 = {}
        self.vector = []

    def __str__(self):
        return f"Node: {self.election_id}"

    def add_child(self, child):
        child.parent = self
        self.children.append(child)
        self.leaf = False


def _generate_tree(
    num_nodes: int, num_internal_nodes: int, rng
):
    """
    Generates a random tree following the algorithm of `Alonso, Rémy, Schott (1997)
    <https://link.springer.com/article/10.1007/BF02522824>`_
    """

    sequence = []
    sizes = []
    larges = []
    ctr = 0
    inner_ctr = 0

    # Mix the patterns, the patterns are defined in Uniform generation of a Schröder tree
    patterns = ["M0" for _ in range(num_nodes - num_internal_nodes)] + [
        "M1" for _ in range(num_internal_nodes)
    ]
    rng.shuffle(patterns)

    # Replace the patterns by their representation as words
    for i, pattern in enumerate(patterns):
        if pattern == "M0":
            sequence.append("x" + str(ctr))
            sizes.append(1)
            ctr += 1
        elif pattern == "M1":
            sequence.append("v" + str(inner_ctr))
            sequence.append("()1")  # semi_edge, instead of 'o'
            sequence.append("f1")  # multi_edge
            sequence.append("f1")  # multi_edge
            sizes.append(4)
            larges.append(i)
            inner_ctr += 1

    # Add missing edges
    # TODO: opposite between semi_edge and multi_edge?
    num_semi_edges = 2 * num_internal_nodes
    num_multi_edges = num_internal_nodes
    edges_to_position = num_nodes - 1 - num_semi_edges  # 1 tree, no classical edges

    possible_insert_pos = []
    for i, elem in enumerate(sequence):
        if elem == "()1":
            possible_insert_pos.append(i)
    insert_pos_indices = list(
        rng.choice(range(len(possible_insert_pos)), size=edges_to_position, replace=True)
    )
    insert_pos_indices.sort(reverse=True)  # Inserting from last first to maintain positions
    for index in insert_pos_indices:
        sizes[larges[index]] += 1
        sequence.insert(possible_insert_pos[index], "f1")

    for i in range(len(possible_insert_pos)):
        sequence.remove("()1")

    return sequence, sizes


def _turn_pattern_into_tree(pattern) -> GroupSeparableNode:
    """
    Converts a pattern into a tree.
    """
    stack = []
    for i, element in enumerate(pattern):
        if "x" in element or "v" in element:
            stack.append(GroupSeparableNode(element))
        if "f" in element:
            parent = stack.pop()
            child = stack.pop()
            parent.add_child(child)
            stack.append(parent)
    return stack[0]


def _cycle_lemma(sequence):
    pos = 0
    height = 0
    min_height = 0
    pos_min = 0
    for element in sequence:
        if "x" in element or "v" in element:
            if height <= min_height:
                pos_min = pos
                min_height = height
            height += 1
        if "f" in element:
            height -= 1
        pos += 1

    # TODO: I don't understand why there is no randomness here

    # rotate
    for _ in range(pos_min):
        element = sequence.pop(0)
        sequence.append(element)

    return sequence


def _add_num_leaf_descendants(node):
    """add total number of descendants to each internal node"""

    if node.leaf:
        node.num_leaf_descendants = 1
    else:
        node.num_leaf_descendants = 0
        for child in node.children:
            node.num_leaf_descendants += _add_num_leaf_descendants(child)

    return node.num_leaf_descendants


def _add_scheme(node):
    for starting_pos in node.scheme_1:
        pos = starting_pos
        for child in node.children:
            if pos in child.scheme_1:
                child.scheme_1[pos] += node.scheme_1[starting_pos]
            else:
                child.scheme_1[pos] = node.scheme_1[starting_pos]
            pos += child.num_leaf_descendants

    for starting_pos in node.scheme_2:
        pos = starting_pos
        for child in node.children:
            if pos in child.scheme_2:
                child.scheme_2[pos] += node.scheme_2[starting_pos]
            else:
                child.scheme_2[pos] = node.scheme_2[starting_pos]
            pos -= child.num_leaf_descendants

    if node.leaf:
        _construct_vector_from_scheme(node)
    else:
        for child in node.children:
            _add_scheme(child)


def _construct_vector_from_scheme(node):
    x = node.scheme_1
    y = node.scheme_2
    node.scheme = {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)}

    weight = 1.0 / sum(node.scheme.values())

    node.vector = [0 for _ in range(GroupSeparableNode.total_num_leaf_descendants)]
    for key in node.scheme:
        node.vector[int(key)] += node.scheme[key] * weight


def _generate_caterpillar(num_leaves: int) -> GroupSeparableNode:
    """
    Generates a caterpillar tree.
    """
    try:
        int(num_leaves)
    except TypeError:
        raise TypeError("The number of leaves needs to be an integer.")
    if int(num_leaves) != num_leaves:
        raise ValueError("The number of leaves should be an integer.")
    if num_leaves <= 0:
        raise ValueError("The number of leaves needs to be 1 or more.")
    num_leaves = int(num_leaves)
    root = GroupSeparableNode("root")
    tmp_root = root
    ctr = 0

    while num_leaves > 2:
        leaf = GroupSeparableNode("x" + str(ctr))
        inner_node = GroupSeparableNode("v" + str(ctr))
        tmp_root.add_child(leaf)
        tmp_root.add_child(inner_node)
        tmp_root = inner_node
        num_leaves -= 1
        ctr += 1

    leaf_1 = GroupSeparableNode("x" + str(ctr))
    leaf_2 = GroupSeparableNode("x" + str(ctr + 1))
    tmp_root.add_child(leaf_1)
    tmp_root.add_child(leaf_2)

    return root


def _generate_balanced_tree(num_leaves: int) -> GroupSeparableNode:
    """
    Generates a balanced tree.
    """
    try:
        int(num_leaves)
    except TypeError:
        raise TypeError("The number of leaves needs to be an integer.")
    if int(num_leaves) != num_leaves:
        raise ValueError("The number of leaves should be an integer.")
    if num_leaves <= 0:
        raise ValueError("The number of leaves needs to be 1 or more.")
    num_leaves = int(num_leaves)
    root = GroupSeparableNode("root")
    ctr = 0

    q = queue.Queue()
    q.put(root)

    while q.qsize() * 2 < num_leaves:
        tmp_root = q.get()
        for _ in range(2):
            inner_node = GroupSeparableNode("v" + str(ctr))
            tmp_root.add_child(inner_node)
            q.put(inner_node)
            ctr += 1

    ctr = 0
    while ctr < num_leaves:
        tmp_root = q.get()
        for _ in range(2):
            node = GroupSeparableNode("x" + str(ctr))
            tmp_root.add_child(node)
            ctr += 1

    return root
