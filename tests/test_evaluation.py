import numpy as np
import pytest

from neuralnaca.evaluation import (
    bootstrap_ratio,
    compare,
    error_metrics,
    regime_breakdown,
    wilcoxon_signed_rank,
)


def test_error_metrics_against_hand_computation():
    y = np.array([[0.5, np.log(0.01), -0.05], [1.0, np.log(0.02), -0.10]])
    p = y + np.array([[0.1, 0.0, 0.01], [-0.3, np.log(1.1), 0.0]])
    m = error_metrics(p, y)
    assert m["CL_MAE"] == pytest.approx(0.2)
    assert m["CL_RMSE"] == pytest.approx(np.sqrt((0.01 + 0.09) / 2))
    assert m["CL_MedAE"] == pytest.approx(0.2)
    assert m["lnCD_MAE"] == pytest.approx(np.log(1.1) / 2)
    assert m["CD_rel_pct"] == pytest.approx(5.0)
    assert m["CD_MAE"] == pytest.approx(0.001)
    assert m["CM_MAE"] == pytest.approx(0.005)


def test_error_metrics_validates_input():
    with pytest.raises(ValueError):
        error_metrics(np.zeros((3, 2)), np.zeros((3, 2)))
    with pytest.raises(ValueError):
        error_metrics(np.full((2, 3), np.nan), np.zeros((2, 3)))


def test_bootstrap_ratio_properties(rng):
    e = np.abs(rng.normal(size=(500, 3)))
    r, lo, hi = bootstrap_ratio(e, e)
    np.testing.assert_allclose([r, lo, hi], 1.0)
    r, lo, hi = bootstrap_ratio(2 * e, e)                           # exact scaling -> degenerate interval
    np.testing.assert_allclose([r, lo, hi], 2.0)
    a = np.abs(rng.normal(size=(500, 3)))
    r, lo, hi = bootstrap_ratio(a, e)
    assert (lo <= r).all() and (r <= hi).all()
    np.testing.assert_array_equal(bootstrap_ratio(a, e, seed=3)[1], bootstrap_ratio(a, e, seed=3)[1])
    r2, _, _ = bootstrap_ratio(a, e, stat="rms")
    np.testing.assert_allclose(r2, np.sqrt((a ** 2).mean(0) / (e ** 2).mean(0)))
    with pytest.raises(ValueError):
        bootstrap_ratio(a, e, stat="median")
    with pytest.raises(ValueError):
        bootstrap_ratio(a, e[:10])


def test_bootstrap_interval_has_nominal_coverage(rng):
    """Coverage check: the 95 % interval contains the true ratio of means in roughly 95 % of repeated experiments."""
    hits, true_ratio = 0, 0.8
    for k in range(100):
        g = np.random.default_rng(k)
        b = g.exponential(1.0, (300, 1))
        a = g.exponential(true_ratio, (300, 1))
        _, lo, hi = bootstrap_ratio(a, b, n_boot=400, seed=k)
        hits += int(lo[0] <= true_ratio <= hi[0])
    assert 85 <= hits <= 100


def test_wilcoxon_detects_systematic_differences(rng):
    b = np.abs(rng.normal(size=(300, 3)))
    assert (wilcoxon_signed_rank(0.7 * b, b) < 1e-10).all()
    assert (wilcoxon_signed_rank(b + rng.normal(0, 1e-3, b.shape), b) > 1e-3).all()


def test_regime_breakdown_partitions_the_points(rng):
    from neuralnaca.domain import sample_points

    pts = sample_points(1000, "random", 1)
    e = np.abs(rng.normal(size=(1000, 3)))
    rows = regime_breakdown(pts, e, e)
    for var in ("alpha", "Re"):
        assert sum(r["n"] for r in rows if r["variable"] == var) == 1000


def test_compare_structure(rng):
    from neuralnaca.domain import sample_points

    y = rng.normal(size=(200, 3))
    pts = sample_points(200, "random", 0)
    out = compare(y + 0.01, {"ref": y + 0.02}, y, pts, n_boot=200)
    assert out["n_test"] == 200 and set(out["metrics"]) == {"model", "ref"}
    vs = out["vs"]["ref"]
    assert vs["mae_ratio"]["CL"][0] == pytest.approx(0.5)
    assert vs["win_fraction"]["CL"] == 1.0
    assert {"mae_ratio", "rmse_ratio", "wilcoxon_p", "win_fraction", "breakdown"} <= set(vs)
