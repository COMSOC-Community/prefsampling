from __future__ import annotations

import numpy as np

from itertools import permutations, combinations_with_replacement, product

from prefsampling.inputvalidators import validate_int
from prefsampling.tree.node import Node
from prefsampling.combinatorics import comb


def validate_num_leaves_nodes(num_leaves: int, num_internal_nodes: int | None):
    """
    Validates the number of leaves of a Schröder tree based on the number of internal nodes.
    Raises ValueError if something goes wrong.
    """
    validate_int(num_leaves, "number of leaves", lower_bound=1)
    if num_internal_nodes is not None:
        validate_int(num_internal_nodes, "number of internal nodes", lower_bound=0)
        if num_internal_nodes == 0 and num_leaves != 1:
            raise ValueError(
                "If the number of internal nodes is 0, there can only be one leaf."
            )
        if num_internal_nodes > num_leaves - 1:
            raise ValueError(
                "The number of internal nodes cannot exceed the number of leaves "
                "minus 1."
            )


def _random_num_internal_nodes(num_leaves: int, rng: np.random.Generator) -> int:
    """
    Samples a number of internal nodes for a Schröder tree at random. Each number of internal nodes
    is sampled with probability proportional to the number of Schroöder trees with the corresponding
    number of internal nodes.

    Parameters
    ----------
        num_leaves: int
            The number of leaves of the tree
        rng: np.random.Generator
            The numpy random number generator.

    Returns
    -------
        int
            The number of internal nodes
    """
    distribution = np.zeros(num_leaves - 1)
    for i in range(num_leaves - 1):
        distribution[i] = _num_schroeder_tree(i + 1, num_leaves)
    distribution /= distribution.sum()
    return rng.choice(num_leaves - 1, p=distribution) + 1


def _num_schroeder_tree(num_internal_nodes, num_leaves):
    return (
        comb(num_leaves - 1, num_internal_nodes)
        * comb(num_leaves - 1 + num_internal_nodes, num_leaves)
        / (num_leaves - 1)
    )


def schroeder_tree(
    num_leaves: int, num_internal_nodes: int | None = None, seed: int = None
) -> Node:
    """
    Generates a random Schöder tree with suitable number of leaves and internal nodes.
    The algorithm was presented by `Alonso, Rémy, Schott (1997a)
    <https://www.sciencedirect.com/science/article/pii/S0020019097001749>`_ building on `Alonso,
    Rémy, Schott (1997b) <https://link.springer.com/article/10.1007/BF02522824>`_.

    If the number of internal node is not give, it is selected at random such that each number of
    internal nodes is sampled with probability proportional to the number of Schroöder trees with
    the corresponding number of internal nodes.

    Parameters
    ----------
        num_leaves: int
            The number of leaves of the tree
        num_internal_nodes: int | None
            The number of internal nodes of the tree
        seed : int, default: :code:`None`
            Seed for numpy random number generator

    Returns
    -------
        Node
            The root of the tree

    Validation
    ----------

        This sampler is supposed to yield a uniform distribution, but does not in practice.

        .. image:: ../validation_plots/tree/schroeder.png
          :width: 800
          :alt: Observed versus theoretical frequencies for the Schröder tree sampler

    References
    ----------

        `Uniform generation of a Schröder tree
        <https://www.sciencedirect.com/science/article/pii/S0020019097001749>`_,
        *L. Alonso, J.L. Rémy and R. Schott*,
        Information Processing Letters, Volume 64, Issue 6, 1997.

        `A linear-time algorithm for the generation of trees
        <https://link.springer.com/article/10.1007/BF02522824>`_,
        *L. Alonso, J.L. Rémy and R. Schott*,
        Algorithmica 17, 162–182, 1997
    """
    validate_num_leaves_nodes(num_leaves, num_internal_nodes)

    if num_internal_nodes == 0 or num_leaves == 1:
        return Node(0)

    rng = np.random.default_rng(seed)

    if num_internal_nodes is None:
        num_internal_nodes = _random_num_internal_nodes(num_leaves, rng)

    # Mix the patterns, the patterns are defined in "Uniform generation of a Schröder tree"
    # where bold edges seem to indicate multi-edges (and not semi-ones as in the paper)
    patterns = ["M0" for _ in range(num_leaves)] + [
        "M1" for _ in range(num_internal_nodes)
    ]
    rng.shuffle(patterns)

    # Replace the patterns by their representation as words
    sequence = []
    leaf_ctr = 0
    internal_ctr = 0
    possible_insert_pos = []
    for i, pattern in enumerate(patterns):
        if pattern == "M0":
            sequence.append("x" + str(leaf_ctr))
            leaf_ctr += 1
        elif pattern == "M1":
            sequence.append("v" + str(internal_ctr))
            possible_insert_pos.append(len(sequence))
            sequence.append("()1")  # multi_edge using '()' instead of 'o'
            sequence.append("f1")  # semi_edge
            sequence.append("f1")  # semi_edge
            internal_ctr += 1

    # Add missing edges (p = 1 tree, e = 0 classical edges)
    edges_to_position = num_leaves - 1 - num_internal_nodes
    insert_pos_indices = list(
        rng.choice(
            range(len(possible_insert_pos)), size=edges_to_position, replace=True
        )
    )
    insert_pos_indices.sort(reverse=True)  # Last to first to maintain positions
    for index in insert_pos_indices:
        sequence.insert(possible_insert_pos[index], "f1")
    sequence = [s for s in sequence if s != "()1"]

    # Apply the cycle lemma, since p = 1 (a single tree), there is only one permutation
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
        elif "f" in element:
            height -= 1
        else:
            raise ValueError(
                f"There should not be an element {element} at this point..."
            )
        pos += 1
    # Apply rotation, pos_min indicates the position of the last node with minimum height
    sequence = sequence[pos_min:] + sequence[:pos_min]

    # Transform the sequence into a tree
    nodes = []
    for i, element in enumerate(sequence):
        if "x" in element or "v" in element:
            nodes.append(Node(int(element.replace("x", "").replace("v", ""))))
        if "f" in element:
            parent = nodes.pop()
            child = nodes.pop()
            parent.add_child(child)
            nodes.append(parent)

    return nodes[0]


def schroeder_tree_lescanne(
    num_leaves: int, num_internal_nodes: int | None = None, seed: int = None
) -> Node:
    """
    Samples a random Schröder tree following the algorithm provided by `Lescanne (2022)
    <https://arxiv.org/abs/2205.11982>`_. The sampler is directly taken from the `corresponding
    GitHub repository <https://github.com/PierreLescanne/Motzkin>`_ (no licence available).

    If a specific number of internal nodes is given, trees are sampled at random until the required
    number of internal nodes is achieved (rejection sampling).

    Parameters
    ----------
        num_leaves: int
            The number of leaves of the tree
        num_internal_nodes: int | None
            The number of internal nodes of the tree
        seed : int, default: :code:`None`
            Seed for numpy random number generator

    Returns
    -------
        Node
            The root of the tree

    Validation
    ----------

        This sampler has unknown probability distribution but is efficient for large numbers of
        leaves.

        .. image:: ../validation_plots/tree/schroeder_lescanne.png
          :width: 800
          :alt: Observed versus theoretical frequencies for the Schröder tree sampler by Lescanne

    References
    ----------

        `Holonomic equations and efficient random generation of binary trees
        <https://dmtcs.episciences.org/12920>`_,
        *Pierre Lesscanne*,
        Discrete Mathematics & Theoretical Computer Science, vol. 25:2, 2024.
    """

    def build_tree(
        current_node, left_child_pos, right_child_pos, current_white, to_merge
    ):
        """
        Auxiliary function that converts the result of the algorithm of Lescanne (2022) into
        our class for Nodes and trees.
        """
        if current_white:
            to_merge.append(current_node.identifier)
        for position in left_child_pos, right_child_pos:
            if position < len(tree_data):
                child = Node(tree_data[position][0])
                current_node.add_child(child)
                if child.identifier % 2 == 1:
                    build_tree(
                        child,
                        child.identifier,
                        child.identifier + 1,
                        tree_data[position][1],
                        to_merge,
                    )

    validate_num_leaves_nodes(num_leaves, num_internal_nodes)

    if num_internal_nodes == 0 or num_leaves == 1:
        return Node(0)

    rng = np.random.default_rng(seed)

    # Run the algorithm of Lescanne (2022) to obtain a tree
    tree_data = []
    while True:
        tree_data = [(0, False)]
        for i in range(1, num_leaves):
            while True:
                rand = rng.random()
                x = int(rand * (6 * i - 4))
                k = int(x // 3)
                xmod3 = x % 3
                if xmod3 == 0:  # L1
                    tree_data.append((2 * i, False))
                    tree_data.append(tree_data[k])
                    tree_data[k] = (2 * i - 1, False)
                    break
                elif xmod3 == 1:  # L2
                    if tree_data[k][0] % 2 == 1:  # fst (v[k]) is odd
                        tree_data.append((2 * i, False))
                        tree_data.append((tree_data[k][0], True))
                        tree_data[k] = (2 * i - 1, False)
                        break
                    else:
                        if k % 2 == 1:  # k is odd
                            if tree_data[k + 1][1]:  # the other leaf is white
                                pass
                            else:  # the other leaf is black
                                tree_data.append(tree_data[k])
                                tree_data.append((2 * i, False))
                                tree_data[k] = tree_data[k + 1]
                                tree_data[k + 1] = (2 * i - 1, True)
                                break
                        else:
                            tree_data.append(tree_data[k])
                            tree_data.append((2 * i, False))
                            tree_data[k] = (2 * i - 1, True)
                            break
                elif xmod3 == 2:  # L3
                    tree_data.append(tree_data[k])
                    tree_data.append((2 * i, False))
                    tree_data[k] = (2 * i - 1, False)
                    break
                else:
                    break
        num_legal_white_edges = sum(1 for n in tree_data[1:] if n[0] % 2 == 1 and n[1])
        if (
            not num_internal_nodes
            or len(tree_data) - num_legal_white_edges == num_internal_nodes + num_leaves
        ):
            break

    # Build the tree with our Node class based on the tree data
    root = Node(tree_data[0][0])
    nodes_to_merge = []
    build_tree(root, root.identifier, root.identifier + 1, False, nodes_to_merge)

    # Merge the nodes that need to be merged (the "white edges")
    for identifier in nodes_to_merge:
        root.merge_with_parent(identifier)
    return root


def schroeder_tree_brute_force(
    num_leaves: int, num_internal_nodes: int | None = None, seed: int = None
) -> Node:
    """
    Sample a Schröder tree uniformly at random by enumerating all possible Schröder tree and
    selecting one of them uniformly at random. This procedure is particularly inefficient but
    ensures uniformity.

    Parameters
    ----------
        num_leaves: int
            The number of leaves of the tree
        num_internal_nodes: int | None
            The number of internal nodes of the tree
        seed : int, default: :code:`None`
            Seed for numpy random number generator

    Returns
    -------
        Node
            The root of the tree

    Validation
    ----------

        This sampler yields a uniform probability distribution but enumerates all trees (and
        is thus slow).

        .. image:: ../validation_plots/tree/schroeder_brute_force.png
          :width: 800
          :alt: Observed versus theoretical frequencies for the Schröder tree sampler by brute force

    """
    all_trees = all_schroeder_tree(num_leaves, num_internal_nodes=num_internal_nodes)
    rng = np.random.default_rng(seed)
    res = rng.choice(all_trees)  # type: Node
    res.rename_frontier()
    return res


def _partition_schroeder_nodes(num_nodes: int, num_leaves: int) -> list[list[int]]:
    """
    Returns all partitions of the number of leaves across the number of internal nodes such that
    all internal nodes receive at least two leaves.

    Parameters
    ----------
        num_nodes: int
            Number of internal nodes
        num_leaves: int
            Number of leaves

    Returns
    -------
        list[list[int]]
            List of lists representing the number of leaves per nodes

    """
    res = []
    num_leaves -= 2 * num_nodes
    for c in combinations_with_replacement(range(num_nodes), num_leaves):
        tmp_res = [2 for _ in range(num_nodes)]
        for i in c:
            tmp_res[i] += 1
        res.append(tmp_res)
    return res


def all_schroeder_tree(num_leaves: int, num_internal_nodes: int = None) -> list[Node]:
    """
    Reruns all Schröder trees with given numbers of leaves and internal nodes.

    Parameters
    ----------
        num_leaves: int
            Number of leaves
        num_internal_nodes: int
            Number of internal nodes

    Returns
    -------
        list[Node]
            List of the tree roots
    """

    def aux(leaves_to_place, internal_to_place, counter):
        """
        Recursive generator.
        """
        # print(f"{counter}: {leaves_to_place}, {internal_to_place}")
        if leaves_to_place == 0:
            # print(f"{counter}: fast-tracked yielding leaf")
            leaf = Node(counter)
            leaf.leaf = True
            yield leaf
            return
        if leaves_to_place == 2:
            subtree = Node(counter)
            leaf1 = Node(counter + 1)
            leaf1.leaf = True
            leaf2 = Node(counter + 2)
            leaf2.leaf = True
            subtree.add_child(leaf1)
            subtree.add_child(leaf2)
            # print(f"{counter}: fast-tracked yielding tree")
            yield subtree
            return

        for leaves in range(leaves_to_place + 1):
            for internal in range(
                max(2 - leaves, 0), min(leaves_to_place // 2 + 1, internal_to_place)
            ):
                if (
                    leaves_to_place - leaves >= internal * 2
                ):  # At least 2 leaves per node
                    # print(f"{counter}\tleaves = {leaves}, internal = {internal}")
                    # print(f"{counter}\tpartitions = "
                    #       f"{partition_schroeder_nodes(internal, leaves_to_place - leaves)}")
                    for children_part in _partition_schroeder_nodes(
                        internal, leaves_to_place - leaves
                    ):
                        # print(f"{counter}\t\tchildren_part={children_part}")
                        for children_perm in set(
                            permutations([0 for _ in range(leaves)] + children_part)
                        ):
                            # print(f"{counter}\t\t\tchildren_perm={children_perm}")
                            current_counter = counter

                            child_generators = []
                            for num_child in children_perm:
                                child_generators.append(
                                    aux(
                                        num_child,
                                        internal_to_place - internal,
                                        current_counter,
                                    )
                                )
                                current_counter += max(num_child, 1)

                            for combination in product(*child_generators):
                                current_node = Node(current_counter)
                                if not combination:
                                    current_node.leaf = True
                                for child in combination:
                                    current_node.add_child(child)

                                # print(f"{counter}\t\t\t({leaves}, {internal}) yielding: "
                                #       f"{current_node.anonymous_tree_representation()}")
                                yield current_node

    validate_num_leaves_nodes(num_leaves, num_internal_nodes)

    if num_internal_nodes == 0 or num_leaves == 1:
        return [Node(0)]

    if num_internal_nodes is None:
        num_internal_nodes = range(1, num_leaves)
    else:
        num_internal_nodes = [num_internal_nodes]
    outcome = []
    tree_repr = set()
    for k in num_internal_nodes:
        for root in aux(num_leaves, k, 0):
            if root.anonymous_tree_representation() not in tree_repr:
                if root.num_internal_nodes() == k:
                    outcome.append(root)
                    tree_repr.add(root.anonymous_tree_representation())
    return outcome
