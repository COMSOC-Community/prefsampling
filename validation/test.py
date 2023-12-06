import numpy as np

from prefsampling.ordinal.groupseparable import _number_decomposition_tree, _generate_tree, \
    _cycle_lemma

rng = np.random.default_rng()
num_candidates = 3

count = {}

for _ in range(100000):

    buckets = np.zeros(num_candidates - 1)
    for r in range(1, num_candidates):
        buckets[r - 1] = _number_decomposition_tree(num_candidates, r)
    buckets /= buckets.sum()
    num_internal_nodes = rng.choice(len(buckets), p=buckets) + 1

    num_nodes = num_candidates + num_internal_nodes
    seq, _ = _generate_tree(num_nodes, num_internal_nodes, rng)

    seq = _cycle_lemma(seq)

    if str(seq) in count:
        count[str(seq)] += 1
    else:
        count[str(seq)] = 1

print(count.values())
