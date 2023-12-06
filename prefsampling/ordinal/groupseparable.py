import math
import queue
from itertools import chain

import numpy as np

from prefsampling.decorators import validate_num_voters_candidates


@validate_num_voters_candidates
def group_separable(
    num_voters: int, num_candidates: int, tree: str = "random", seed: int = None
):
    """
    Algorithm from
    'The Complexity of Election Problems with Group-Separable Preferences' paper.
    """

    rng = np.random.default_rng(seed)

    while True:
        m = num_candidates
        n = num_voters

        if tree == "random":
            func = (
                lambda m, r: 1.0
                / (m - 1)
                * math.comb(m - 1, r)
                * math.comb(m - 1 + r, m)
            )
            buckets = [func(m, r) for r in range(1, m)]

            denominator = sum(buckets)
            buckets = [buckets[i] / denominator for i in range(len(buckets))]

            num_internal_nodes = rng.choice(len(buckets), 1, p=buckets)[0] + 1

            decomposition_tree = _decompose_tree(
                num_candidates, num_internal_nodes, rng
            )

        elif tree == "caterpillar":
            decomposition_tree = _caterpillar(m)

        elif tree == "balanced":
            decomposition_tree = _balanced(m)
        else:
            raise ValueError(
                f"Incorrect tree type. No such tree type as {tree}."
                f"The available tree types are: random, caterpillar, balanced."
            )

        all_inner_nodes = _get_all_inner_nodes(decomposition_tree)

        votes = []
        for _ in range(n):
            signature = [rng.choice([0, 1]) for _ in range(len(all_inner_nodes))]

            for i, node in enumerate(all_inner_nodes):
                node.reverse = signature[i]

            raw_vote = _sample_a_vote(decomposition_tree)
            vote = [int(candidate.replace("x", "")) for candidate in raw_vote]
            votes.append(vote)

            for i, node in enumerate(all_inner_nodes):
                node.reverse = False

        return votes


def _decompose_tree(num_leaves, num_internal_nodes, rng):
    """
    Algorithm from
    'Uniform generation of a Schroder tree' paper.
    """

    num_nodes = num_leaves + num_internal_nodes

    patterns = _generate_patterns(num_nodes, num_internal_nodes, rng)
    seq, sizes = _generate_tree(num_nodes, num_internal_nodes, patterns, rng)

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
        return self.election_id

    def add_child(self, child):
        child.parent = self
        self.children.append(child)
        self.leaf = False


def _generate_patterns(num_nodes: int = None, num_internal_nodes: int = None, rng=None):
    """
    Generates a random patter.
    """
    patterns = ["M0" for _ in range(num_nodes - num_internal_nodes)] + [
        "M1" for _ in range(num_internal_nodes)
    ]
    rng.shuffle(patterns)
    return patterns


def _generate_tree(
    num_nodes: int = None, num_internal_nodes: int = None, patterns=None, rng=None
):
    """
    Algorithm from
    'A linear-time embedding_id for the generation of trees' paper.
    """

    sequence = []
    sizes = []
    larges = []
    ctr = 0
    inner_ctr = 0
    for i, pattern in enumerate(patterns):
        if pattern == "M0":
            sequence.append("x" + str(ctr))
            sizes.append(1)
            ctr += 1
        elif pattern == "M1":
            sequence.append("v" + str(inner_ctr))
            sequence.append("()1")  # instead of 'o'
            sequence.append("f1")
            sequence.append("f1")
            sizes.append(4)
            larges.append(i)
            inner_ctr += 1

    num_classical_edges = 0
    num_semi_edges = 2 * num_internal_nodes
    num_multi_edges = num_internal_nodes
    num_trees = 1
    pos = 1
    num_edges = num_nodes - num_trees - num_semi_edges - num_classical_edges

    pos_to_insert = []
    for i, elem in enumerate(sequence):
        if elem == "()1":
            pos_to_insert.append(i)

    choices = list(
        rng.choice([i for i in range(len(pos_to_insert))], size=num_edges, replace=True)
    )
    choices.sort(reverse=True)

    for choice in choices:
        sizes[larges[choice]] += 1
        sequence.insert(pos_to_insert[choice], "f1")

    for i in range(len(pos_to_insert)):
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
    min = 0
    pos_min = 0
    for element in sequence:
        if "x" in element or "v" in element:
            if height <= min:
                pos_min = pos
                min = height
            height += 1
        if "f" in element:
            height -= 1
        pos += 1

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


def _caterpillar(num_leaves) -> GroupSeparableNode:
    """
    Generates a caterpillar tree.
    """
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


def _balanced(num_leaves) -> GroupSeparableNode:
    """
    Generates a balanced tree.
    """
    root = GroupSeparableNode("root")
    ctr = 0

    q = queue.Queue()
    q.put(root)

    # while ctr < num_leaves-2:
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
