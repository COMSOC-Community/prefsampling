from copy import copy


class Node:

    def __init__(self, element_id):
        self.element_id = element_id
        self.parent = None
        self.children = []
        self.leaf = True
        self.reverse = False

    def __str__(self):
        return f"Node: {self.element_id}"

    def __repr__(self):
        return f"Node: {self.element_id}"

    def add_child(self, child):
        child.parent = self
        self.children.append(child)
        self.leaf = False

    def tree_representation(self):
        if len(self.children) == 1:
            return self.children[0].tree_representation()
        s = f"{self.element_id}("
        s += ', '.join(n.tree_representation() for n in self.children)
        s += ")"
        return s

    def anonymous_tree_representation(self):
        if len(self.children) == 1:
            return self.children[0].anonymous_tree_representation()
        s = f"_("
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
