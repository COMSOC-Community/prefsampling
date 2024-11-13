from unittest import TestCase

from prefsampling.tree.node import Node
from prefsampling.tree.schroeder import all_schroeder_tree, _num_schroeder_tree


class TestTree(TestCase):
    def test_node(self):
        root = Node(0)
        internal_node = Node(1)
        leaf1 = Node(2)
        leaf2 = Node(3)
        leaf3 = Node(4)
        root.add_child(internal_node)
        root.add_child(leaf1)
        internal_node.add_child(leaf2)
        internal_node.add_child(leaf3)

        root.__str__()
        root.__repr__()

        # Basic methods
        self.assertEqual(root.get_child(1), internal_node)
        self.assertEqual(root.get_child("AZE"), None)
        self.assertEqual(root.num_leaves(), 3)
        self.assertEqual(root.num_internal_nodes(), 2)
        self.assertEqual(root.internal_nodes(), [root, internal_node])

        # Test for Schröder trees
        self.assertTrue(root.is_schroeder())
        new_root = Node(0)
        new_root.add_child(Node(1))
        self.assertFalse(new_root.is_schroeder())

        # Tree copy
        root_copy = root.copy_tree()
        self.assertEqual(root_copy.identifier, 0)
        self.assertEqual(len(root_copy.children), 2)
        self.assertEqual(set(c.identifier for c in root_copy.children), {1, 2})
        self.assertEqual(len(root_copy.get_child(1).children), 2)
        self.assertEqual(
            set(c.identifier for c in root_copy.get_child(1).children), {3, 4}
        )

        # Frontier renaming
        root.rename_frontier()
        self.assertEqual(root.children[0].children[0].identifier, 0)
        self.assertEqual(root.children[0].children[1].identifier, 1)
        self.assertEqual(root.children[1].identifier, 2)
        root.rename_frontier(new_names=["a", "b", "c"])
        self.assertEqual(root.children[0].children[0].identifier, "a")
        self.assertEqual(root.children[0].children[1].identifier, "b")
        self.assertEqual(root.children[1].identifier, "c")
        with self.assertRaises(ValueError):
            root.rename_frontier(new_names=["a"])

    def test_all_schroeder_trees(self):
        schroeder_numbers = {
            2: 1,
            3: 3,
            4: 11,
            5: 45,
            6: 197,
            7: 903,
            # 8: 4279,
            # 9: 20793,
            # 10: 103049,
            # 11: 518859,
            # 12: 2646723
        }

        # Check that the right number of trees is returned
        for num_leaves, count in schroeder_numbers.items():
            self.assertTrue(
                len(
                    set(
                        t.anonymous_tree_representation()
                        for t in all_schroeder_tree(num_leaves)
                    )
                )
                == count
            )
            c = 0
            for num_internal_nodes in range(1, num_leaves):
                tmp_c = len(
                    set(
                        t.anonymous_tree_representation()
                        for t in all_schroeder_tree(num_leaves, num_internal_nodes)
                    )
                )
                self.assertTrue(
                    _num_schroeder_tree(num_internal_nodes, num_leaves) == tmp_c
                )
                c += tmp_c
            self.assertTrue(c == count)

        # Check that trees are always returned in the same order:
        r = [s.tree_representation() for s in all_schroeder_tree(7)]
        for _ in range(200):
            self.assertTrue(
                r == [s.tree_representation() for s in all_schroeder_tree(7)]
            )
