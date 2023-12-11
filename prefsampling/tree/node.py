class Node:

    def __init__(self, identifier):
        self.identifier = identifier
        self.parent: Node = None
        self.children: list[Node] = []
        self.leaf: bool = True
        self.reverse: bool = False

    def __str__(self) -> str:
        return f"Node: {self.identifier}"

    def __repr__(self) -> str:
        return f"Node: {self.identifier}"

    def add_child(self, child):
        child.parent = self
        self.children.append(child)
        self.leaf = False

    def get_child(self, identifier):
        if self.identifier == identifier:
            return self
        for c in self.children:
            result = c.get_child(identifier)
            if result is not None:
                return result
        return None

    def num_leaves(self):
        if self.leaf:
            return 1
        res = 0
        for c in self.children:
            res += c.num_leaves()
        return res

    def num_internal_nodes(self):
        if self.leaf:
            return 0
        res = 1
        for c in self.children:
            res += c.num_internal_nodes()
        return res

    def shallow_copy_node(self):
        res = Node(self.identifier)
        res.leaf = self.leaf
        res.reverse = self.reverse
        return res

    def copy_tree(self, tree: dict = None):
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

    def rename_frontier(self, new_names=None):
        stack = [self]
        leaf_counter = 0

        while stack:
            current_node = stack.pop()

            if current_node.leaf:
                if new_names is None:
                    current_node.identifier = leaf_counter
                else:
                    current_node.identifier = new_names[leaf_counter]
                leaf_counter += 1

            stack.extend(reversed(current_node.children))

    def tree_representation(self):
        if len(self.children) == 1:
            return self.children[0].tree_representation()
        s = f"{self.identifier}("
        s += ', '.join(n.tree_representation() for n in self.children)
        s += ")"
        return s

    def anonymous_tree_representation(self):
        if len(self.children) == 0:
            return "_"
        s = f"{len(self.children)}("
        s += ', '.join(n.anonymous_tree_representation() for n in self.children)
        s += ")"
        return s

    def get_all_inner_nodes(self):
        if self.leaf:
            return []
        output = [[self]]
        for c in self.children:
            output.append(c.get_all_inner_nodes())
        return [n for sublist in output for n in sublist]
