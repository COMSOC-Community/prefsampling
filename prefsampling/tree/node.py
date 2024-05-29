from __future__ import annotations


class Node:
    """
    Class used to represent nodes in a tree.

    Parameters
    ----------
        identifier : str | int
            The identifier of the node.

    Attributes
    ----------
        identifier : str | int
            The identifier of the node.
        parent : Node | None
            The parent of the node.
        children : list[Node]
            The children list of the node. The order is important.
        leaf : bool
            A boolean indicating whether the node is a leaf or not. Should always be equivalent
            to :code:`len(self.children) == 0`.
        reverse : bool
            A boolean used when sampling group separable preferences.
    """

    def __init__(self, identifier: str | int):
        self.identifier: str | int = identifier
        self.parent: Node | None = None
        self.children: list[Node] = []
        self.leaf: bool = True
        self.reverse: bool = False

    def __str__(self) -> str:
        return f"Node: {self.identifier}"

    def __repr__(self) -> str:
        return f"Node: {self.identifier}"

    def add_child(self, child: Node):
        """
        Add a child to the node and update the `leaf` attribute accordingly.

        Parameters
        ----------
            child: Node
                The child to add.
        """
        child.parent = self
        self.children.append(child)
        self.leaf = False

    def get_child(self, identifier: str | int) -> Node | None:
        """
        Traverses the tree rooted in the node looking for a node with the given identifier. The
        node itself can be returned. If no suitable node is found, :code:`None` is returned.

        Parameters
        ----------
            identifier: str | int
                The identifier of the looked for node.

        Returns
        -------
            Node | None
                A node with the identifier, or None if no such node is found.
        """
        if self.identifier == identifier:
            return self
        for c in self.children:
            result = c.get_child(identifier)
            if result is not None:
                return result
        return None

    def num_leaves(self) -> int:
        """
        Counts the number of leaves of the tree rooted in the node.

        Returns
        -------
            int
                Number of leaves.

        """
        if self.leaf:
            return 1
        res = 0
        for c in self.children:
            res += c.num_leaves()
        return res

    def internal_nodes(self, current_list=None) -> list[Node]:
        if current_list is None:
            current_list = []
        if not self.leaf:
            current_list.append(self)
        for child in self.children:
            child.internal_nodes(current_list)
        return current_list

    def num_internal_nodes(self) -> int:
        """
        Counts the number of internal nodes of the tree rooted in the node.

        Returns
        -------
            int
                Number of internal nodes.

        """
        if self.leaf:
            return 0
        return 1 + sum(c.num_internal_nodes() for c in self.children)

    def merge_with_parent(self, identifier) -> None:
        if self.identifier == identifier:
            self.parent.children.remove(self)
            for c in self.children:
                self.parent.add_child(c)
            self.children = []
        else:
            for c in self.children:
                c.merge_with_parent(identifier)

    def is_schroeder(self) -> bool:
        if self.leaf:
            return True
        if len(self.children) == 1:
            return False
        return all(c.is_schroeder() for c in self.children)

    def shallow_copy_node(self) -> Node:
        """
        Copies the node without copying its children.

        Returns
        -------
            Node
                The copy of the node.

        """
        res = Node(self.identifier)
        res.leaf = self.leaf
        res.reverse = self.reverse
        return res

    def copy_tree(self, tree: dict = None) -> Node:
        """
        Copies the tree rooted in the node. All nodes are copied and the children are attached
        where they should be.

        Returns
        -------
            Node
                The copy of the root node.

        """
        if tree is None:
            tree = {}
        node_copy = tree.get(self.identifier, None)
        if node_copy is not None:
            return node_copy

        node_copy = self.shallow_copy_node()
        tree[self.identifier] = node_copy

        for c in self.children:
            child_copy = c.copy_tree(tree)
            node_copy.add_child(child_copy)
        return node_copy

    def rename_frontier(self, new_names: list = None) -> None:
        """
        Renames the frontier of the tree rooted in the node. Leaves are renamed from the left-most
        leaf to the right-most one. If :code:`new_names == None`, leaves are renamed `0, 1, 2...`.

        Parameters
        ----------
            new_names: list, optional
                The new names for the leaves, the list should be as long as the number of leaves.

        Returns
        -------

        """
        stack = [self]
        leaf_counter = 0

        while stack:
            current_node = stack.pop()

            if current_node.leaf:
                if new_names is None:
                    current_node.identifier = leaf_counter
                else:
                    if leaf_counter < len(new_names):
                        current_node.identifier = new_names[leaf_counter]
                    else:
                        raise ValueError(
                            f"The list of new names is not long enough, there are at least "
                            f"{leaf_counter} leaves."
                        )
                leaf_counter += 1

            stack.extend(reversed(current_node.children))

    def tree_representation(self) -> str:
        """
        Returns a string representation of the tree rooted in the node. The nodes are represented
        by their identifier.

        Returns
        -------
            str
                The representation of the tree.
        """
        if len(self.children) == 0:
            return str(self.identifier)
        s = f"{self.identifier}("
        s += ", ".join(n.tree_representation() for n in self.children)
        s += ")"
        return s

    def anonymous_tree_representation(self) -> str:
        """
        Returns a string representation of the tree rooted in the node. Nodes are represented by
        their number of children and leaves by the underscore character `_`.

        Returns
        -------
            str
                The representation of the tree.
        """
        if len(self.children) == 0:
            return "_"
        s = f"{len(self.children)}("
        s += ", ".join(n.anonymous_tree_representation() for n in self.children)
        s += ")"
        return s
