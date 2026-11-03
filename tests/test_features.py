import numpy as np
from scipy import sparse

from abuse_risk.data import YelpData, make_features


def test_graph_aggregation_uses_features_and_structure_only():
    x = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    graph = sparse.csr_matrix([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
    data = YelpData(x, np.array([0, 1, 0]), {"rur": graph})
    first = make_features(data)
    data.y[:] = 1 - data.y
    second = make_features(data)
    assert first.equals(second)
    assert first.loc[0, "rur_mean_01"] == 4
    assert first.loc[0, "rur_std_01"] == 1
    assert first.loc[1, "rur_mean_01"] == 1
