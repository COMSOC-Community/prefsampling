import math
from copy import deepcopy
from itertools import permutations

import numpy as np

from prefsampling.tree.node import Node


def _random_num_internal_nodes(num_leaves, rng):
    distribution = np.zeros(num_leaves - 1)
    for i in range(num_leaves - 1):
        distribution[i] = _num_schroeder_tree(num_leaves + i + 1, i + 1)
    distribution /= distribution.sum()
    return rng.choice(num_leaves - 1, p=distribution) + 1


def _num_schroeder_tree(n, k):
    return math.comb(n, k) * math.comb(n - 2 - k, n - 1 - 2 * k) / n


def schroeder_tree(num_leaves: int, num_internal_nodes: int = None,
                   seed: int = None) -> Node:
    """
    Generates a random Schöder tree with suitable number of leaves and internal nodes.
    The algorithm was presented by `Alonso, Rémy, Schott (1997a)
    <https://www.sciencedirect.com/science/article/pii/S0020019097001749>`_ building on `Alonso,
    Rémy, Schott (1997b) <https://link.springer.com/article/10.1007/BF02522824>`_.

    If the number of internal node is not give, it is generated in a way that ensures a uniform
    distribution over the Schröder trees.

    Parameters
    ----------
        num_leaves: int
            The number of leaves of the tree
        num_internal_nodes: int
            The number of internal nodes of the tree
        seed : int, default: :code:`None`
            Seed for numpy random number generator

    Returns
    -------
        Node
            The root of the tree

    """
    rng = np.random.default_rng(seed)

    if num_internal_nodes is None:
        num_internal_nodes = _random_num_internal_nodes(num_leaves, rng)

    num_nodes = num_leaves + num_internal_nodes

    sequence = []
    sizes = []
    larges = []
    ctr = 0
    inner_ctr = 0

    # Mix the patterns, the patterns are defined in "Uniform generation of a Schröder tree"
    # where bold edges seem to indicate multi-edges (and not semi-ones as in the paper)
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
            sequence.append("()1")  # multi_edge using '()' instead of 'o'
            sequence.append("f1")  # semi_edge
            sequence.append("f1")  # semi_edge
            sizes.append(4)
            larges.append(i)
            inner_ctr += 1

    # Add missing edges
    num_semi_edges = 2 * num_internal_nodes
    edges_to_position = num_nodes - 1 - num_semi_edges  # p = 1 tree, e = 0 classical edges

    possible_insert_pos = []
    for i, elem in enumerate(sequence):
        if elem == "()1":
            possible_insert_pos.append(i)
    insert_pos_indices = list(
        rng.choice(range(len(possible_insert_pos)), size=edges_to_position, replace=True)
    )
    insert_pos_indices.sort(reverse=True)  # Inserting from last to first to maintain positions
    for index in insert_pos_indices:
        sizes[larges[index]] += 1
        sequence.insert(possible_insert_pos[index], "f1")

    for i in range(len(possible_insert_pos)):
        sequence.remove("()1")

    # Apply the cycle lemma
    # Since p = 1 (a single tree), there is only one permutation in our case
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

    # rotate
    for _ in range(pos_min):
        element = sequence.pop(0)
        sequence.append(element)

    # We transform the sequence into a tree
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


def partition_n(n, num_parts):
    # Generate all partitions of n into num_parts parts
    if num_parts < 2:
        yield (n,)
        return

    for i in range(1, n):
        for p in partition_n(n - i, num_parts - 1):
            yield (i,) + p


def all_schroeder_tree(num_leaves: int):
    def aux(node, tree, all_roots, leaves_to_place, counter):
        print(f"Called on {node} with l={leaves_to_place}")
        if leaves_to_place == 0:
            return
        for leaves in range(leaves_to_place + 1):
            print(f"{node}\tleaves={leaves}, range({max(2 - leaves, 0)}, {leaves_to_place // 2 })")
            for internal in range(max(2 - leaves, 0), leaves_to_place // 2 + 1):
                print(f"{node}\t\tinternal={internal}")
                if leaves_to_place - leaves == internal * 2:
                    print(f"{node}\t\tpartitions={list(partition_n(leaves_to_place - leaves, internal))}")
                    for children_part in partition_n(leaves_to_place - leaves, internal):
                        print(f"{node}\t\t\tchildren_part={children_part}")
                        for perm in set(permutations([0 for _ in range(leaves)] + [size for size in children_part if internal > 0])):
                            current_counter = counter
                            current_tree = deepcopy(tree)
                            print(f"{node}\t\t\t\tperm={perm}")
                            for num_child in perm:
                                print(f"{node}\t\t\t\t\tnum_child={num_child}")
                                child = Node(current_counter)
                                node.add_child(child)
                                current_counter += 1
                                current_tree.append(child)
                                print(current_tree)
                                if num_child > 0:
                                    new_tree = deepcopy(current_tree)
                                    all_roots.append(new_tree[0])
                                    aux(new_tree[-1], new_tree, all_roots, num_child, current_counter)

    root = Node(0)
    outcome = []
    aux(root, [root], outcome, num_leaves, 1)
    return outcome


if __name__ == "__main__":
    for r in all_schroeder_tree(3):
        print(r.tree_representation())
