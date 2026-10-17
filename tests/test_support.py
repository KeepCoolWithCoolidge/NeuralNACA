import numpy as np
import pytest

from neuralnaca.data import XfoilDataset
from neuralnaca.domain import sample_points
from neuralnaca.support import SupportModel, support_features


def toy(n=4096, seed=0):
    """Synthetic 'XFOIL' that fails for alpha > 10 at Re < 3e5."""
    pts = sample_points(n, "sobol", seed)
    fail = (pts[:, 4] > 10) & (pts[:, 3] < 3e5)
    res = np.column_stack([np.ones(n), np.full(n, 0.01), np.zeros(n)])
    res[fail] = np.nan
    return XfoilDataset(pts, res)


def test_features_are_unit_scaled_and_gate_camber_position():
    f = support_features(sample_points(1000, "random", 0))
    assert f.min() >= -1e-9 and f.max() <= 1 + 1e-9
    a = support_features(np.array([[0.0, 0.2, 0.1, 1e6, 0.0]]))
    b = support_features(np.array([[0.0, 0.7, 0.1, 1e6, 0.0]]))
    np.testing.assert_array_equal(a, b)


def test_learns_the_failure_region():
    sup = SupportModel.from_dataset(toy())
    good = np.array([[0.02, 0.4, 0.12, 2e6, 3.0]])
    bad = np.array([[0.02, 0.4, 0.12, 1.2e5, 14.0]])
    assert sup.convergence_probability(good)[0] > 0.9
    assert sup.convergence_probability(bad)[0] < 0.1
    assert sup.supported(good, np.array([0.01]))[0] and not sup.supported(bad, np.array([0.01]))[0]


def test_output_range_guard():
    sup = SupportModel.from_dataset(toy())
    good = np.array([[0.02, 0.4, 0.12, 2e6, 3.0]])
    assert sup.cd_max == pytest.approx(0.01)
    assert not sup.supported(good, np.array([0.02]))[0]


def test_round_trip(tmp_path):
    sup = SupportModel.from_dataset(toy(), k=7)
    sup.save(tmp_path / "s.npz")
    back = SupportModel.load(tmp_path / "s.npz")
    pts = sample_points(50, "random", 3)
    np.testing.assert_array_equal(back.convergence_probability(pts), sup.convergence_probability(pts))
    assert back.k == 7


def test_rejects_bad_input(tmp_path):
    with pytest.raises(ValueError):
        SupportModel(np.zeros((3, 5)), np.ones(3, bool), 0.1)
    with open(tmp_path / "x.npz", "wb") as fh:
        np.savez(fh, header=np.array('{"format": "other"}'), points=np.zeros((20, 5)), converged=np.ones(20, bool))
    with pytest.raises(ValueError):
        SupportModel.load(tmp_path / "x.npz")
