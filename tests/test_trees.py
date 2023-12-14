from unittest import TestCase

from prefsampling.tree.schroeder import all_schroeder_tree, _num_schroeder_tree


class TestTree(TestCase):
    def test_all_schroeder_trees(self):
        schroeder_numbers = {
            2: 1,
            3: 3,
            4: 11,
            5: 45,
            6: 197,
            7: 903,
            8: 4279,
            9: 20793,
            # 10: 103049,
            # 11: 518859,
            # 12: 2646723
        }

        # Check that the right number of trees is returned
        for num_leaves, count in schroeder_numbers.items():
            assert (
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
                assert _num_schroeder_tree(num_internal_nodes, num_leaves) == tmp_c
                c += tmp_c
            assert c == count

        # Check that trees are always returned in the same order:
        r = [s.tree_representation() for s in all_schroeder_tree(7)]
        for _ in range(200):
            assert r == [s.tree_representation() for s in all_schroeder_tree(7)]
