import numpy as np

from abuse_risk.validate_strategy import group_split_indices


def test_group_split_keeps_components_disjoint():
    groups = np.repeat(np.arange(300), 3)
    y = np.tile([0, 1, 0], 300)
    split = group_split_indices(y, groups, seed=201)
    assert sum(len(ids) for ids in split.values()) == len(y)
    assert len(set(np.concatenate(list(split.values())))) == len(y)
    group_sets = [set(groups[ids]) for ids in split.values()]
    assert all(not (group_sets[i] & group_sets[j]) for i in range(5) for j in range(i + 1, 5))
