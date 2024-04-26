from __future__ import annotations

from prefsampling.inputvalidators import validate_int
from prefsampling.tree.node import Node


def caterpillar_tree(num_leaves: int, seed: int = None) -> Node:
    """
    Generates a caterpillar tree. In the special case of :code:`num_leaves == 1` then we output a
    single node even if it is strictly speaking not a caterpillar tree.

    Returns
    -------
        Node
            The root of the tree
    """
    validate_int(num_leaves, "number of leaves", lower_bound=1)
    num_leaves = int(num_leaves)
    root = Node(0)
    if num_leaves == 1:
        return root

    tmp_root = root
    ctr = 1

    while num_leaves > 2:
        leaf = Node(ctr)
        inner_node = Node(ctr)
        tmp_root.add_child(leaf)
        tmp_root.add_child(inner_node)
        tmp_root = inner_node
        num_leaves -= 1
        ctr += 1

    leaf_1 = Node(ctr)
    leaf_2 = Node(ctr + 1)
    tmp_root.add_child(leaf_1)
    tmp_root.add_child(leaf_2)

    return root
