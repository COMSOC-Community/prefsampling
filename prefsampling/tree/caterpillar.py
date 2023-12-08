from prefsampling.inputvalidators import validate_positive_int
from prefsampling.tree.node import Node


def generate_caterpillar_tree(num_leaves: int, seed: int = None) -> Node:
    """
    Generates a caterpillar tree.

    Returns
    -------
        Node
            The root of the tree
    """
    validate_positive_int(num_leaves, "number of leaves")
    num_leaves = int(num_leaves)
    root = Node("root")
    tmp_root = root
    ctr = 0

    while num_leaves > 2:
        leaf = Node("x" + str(ctr))
        inner_node = Node("v" + str(ctr))
        tmp_root.add_child(leaf)
        tmp_root.add_child(inner_node)
        tmp_root = inner_node
        num_leaves -= 1
        ctr += 1

    leaf_1 = Node("x" + str(ctr))
    leaf_2 = Node("x" + str(ctr + 1))
    tmp_root.add_child(leaf_1)
    tmp_root.add_child(leaf_2)

    return root
