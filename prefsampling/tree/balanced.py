from __future__ import annotations

import queue

from prefsampling.inputvalidators import validate_int
from prefsampling.tree.node import Node


def balanced_tree(num_leaves: int, seed: int = None) -> Node:
    """
    Generates a balanced tree.

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
    ctr = 1

    q = queue.Queue()
    q.put(root)

    while ctr < 2*num_leaves-1:
        tmp_root = q.get()
        for _ in range(2):
            inner_node = Node(ctr)
            ctr += 1
            tmp_root.add_child(inner_node)
            q.put(inner_node)

    return root
