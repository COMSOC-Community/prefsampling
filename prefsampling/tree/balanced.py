from __future__ import annotations

import queue

from prefsampling.inputvalidators import validate_int
from prefsampling.tree.node import Node


def balanced_tree(num_leaves: int) -> Node:
    """
    Generates a balanced tree.

    Returns
    -------
        Node
            The root of the tree
    """
    validate_int(num_leaves, "number of leaves", lower_bound=1)
    num_leaves = int(num_leaves)
    root = Node("root")
    ctr = 0

    q = queue.Queue()
    q.put(root)

    while q.qsize() * 2 < num_leaves:
        tmp_root = q.get()
        for _ in range(2):
            inner_node = Node(ctr)
            tmp_root.add_child(inner_node)
            q.put(inner_node)
            ctr += 1

    ctr = 0
    while ctr < num_leaves:
        tmp_root = q.get()
        for _ in range(2):
            node = Node(ctr)
            tmp_root.add_child(node)
            ctr += 1

    return root
