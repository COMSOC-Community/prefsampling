from unittest import TestCase

from prefsampling.tree.schroeder import (
    schroeder_tree,
    schroeder_tree_lescanne,
    schroeder_tree_brute_force,
)


class TestTree(TestCase):
    def is_proper_schroeder_tree(self, sampler, num_leaves, num_internal_nodes):
        root = sampler(num_leaves, num_internal_nodes)
        assert root.is_schroeder()
        assert root.num_leaves() == num_leaves
        if num_internal_nodes:
            assert root.num_internal_nodes() == num_internal_nodes

    def test_schroeder_tree_sampler(self):
        for num_leaves in range(-1, 7):
            for num_internal_nodes in [None] + list(range(-1, num_leaves)):
                for sampler in (
                    schroeder_tree,
                    schroeder_tree_brute_force,
                    schroeder_tree_lescanne,
                ):
                    with self.subTest(
                        sampler=sampler,
                        num_leaves=num_leaves,
                        num_internal_nodes=num_internal_nodes,
                    ):
                        if num_leaves < 1 or (
                            num_internal_nodes is not None and num_internal_nodes < 0
                        ):
                            with self.assertRaises(ValueError):
                                sampler(num_leaves, num_internal_nodes)
                        elif num_internal_nodes == 0 and num_leaves > 1:
                            with self.assertRaises(ValueError):
                                sampler(num_leaves, num_internal_nodes)
                        else:
                            for _ in range(200):
                                self.is_proper_schroeder_tree(
                                    sampler, num_leaves, num_internal_nodes
                                )
