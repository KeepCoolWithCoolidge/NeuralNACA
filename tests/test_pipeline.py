"""End-to-end benchmark pipeline on a synthetic teacher (no XFOIL needed): every experiment and the reproduction
levels 0 and 1 run on tiny problems."""

import json

import numpy as np
import pytest

from neuralnaca import benchmark as b
from neuralnaca.data import XfoilDataset
from neuralnaca.domain import sample_points
from neuralnaca.report import build_report, format_reproduction
from neuralnaca.training import TrainConfig
from neuralnaca.xfoil import XfoilConfig

from ._solvers import analytic_teacher

pytestmark = [pytest.mark.slow, pytest.mark.neuralfoil]

SPEC = b.BenchmarkSpec(n_train=128, n_test=48, train=TrainConfig(ensemble=2, steps=60, width=32, depth=2))


def write_synthetic(layout: b.Layout, seed: int):
    layout.mkdirs()
    cfg = XfoilConfig()
    for split, n, kind, s in (("train", SPEC.n_train, "sobol", seed), ("test", SPEC.n_test, "random", seed + 1000)):
        pts = sample_points(n, kind, s)
        res = np.array([analytic_teacher(p, None) for p in pts])
        res[pts[:, 4] > 14.5] = np.nan                                 # some "unconverged" points
        XfoilDataset(pts, res, {"xfoil": cfg.to_dict()}).save(layout.dataset(split, n, seed, cfg))


@pytest.fixture(scope="module")
def layout(tmp_path_factory):
    lay = b.Layout(tmp_path_factory.mktemp("bench"))
    for seed in (0, 1):
        write_synthetic(lay, seed)
        b.run_main(seed, lay, SPEC, nf_sizes=("xxxlarge",))
    return lay


def test_main_results_are_complete(layout):
    r = json.loads((layout.results / "main_s0.json").read_text(encoding="utf-8"))
    assert r["leakage_test_airfoils_in_train"] == 0
    assert r["reflection_check"]["ok"]
    assert r["datasets"]["test"]["n"] == SPEC.n_test
    assert set(r["comparison"]["metrics"]) == {"model", "NeuralFoil xxxlarge"}
    z = np.load(layout.results / "main_s0_pred.npz")
    assert z["y"].shape == z["model"].shape == z["nf_xxxlarge"].shape


def test_cached_model_is_reused(layout):
    before = (layout.models / "full_k2_n128_s0.npz").stat().st_mtime_ns
    b.run_main(0, layout, SPEC, nf_sizes=("xxxlarge",))
    assert (layout.models / "full_k2_n128_s0.npz").stat().st_mtime_ns == before


def test_tampered_cache_is_not_trusted(tmp_path, monkeypatch):
    """A data file whose points differ from the declared design is regenerated, never silently used."""
    import neuralnaca.data as data_mod

    lay = b.Layout(tmp_path)
    write_synthetic(lay, 0)
    test_path = lay.dataset("test", SPEC.n_test, 0, XfoilConfig())
    tr = XfoilDataset.load(lay.dataset("train", SPEC.n_train, 0, XfoilConfig()))
    tr.subset(slice(0, SPEC.n_test)).save(test_path)                  # test file overwritten with training points
    calls = []

    def fake_run(points, cfg, workers, label):
        calls.append(len(points))
        return np.array([analytic_teacher(p, None) for p in points])

    monkeypatch.setattr(data_mod, "run_xfoil", fake_run)
    b.run_main(0, lay, SPEC, nf_sizes=(), write=False)
    assert calls == [SPEC.n_test]


def test_leakage_guard(tmp_path, monkeypatch):
    """Defence in depth: should a test airfoil ever coincide with a training airfoil, the run is refused."""
    lay = b.Layout(tmp_path)
    write_synthetic(lay, 0)
    monkeypatch.setattr(b, "split_train_test_overlap", lambda tr, te: 3)
    with pytest.raises(RuntimeError, match="share an airfoil"):
        b.run_main(0, lay, SPEC, nf_sizes=())


def test_reproduce_level0_is_exact(layout):
    rep = b.reproduce(0, [0, 1], layout)
    for row in rep["seeds"].values():
        assert row["files_ok"] and row["identical"]
    assert "Reproduction level 0" in format_reproduction(rep)


def test_reproduce_level1_retrains_identically(layout, tmp_path, monkeypatch):
    monkeypatch.setattr(b, "BenchmarkSpec", lambda: SPEC)
    out = b.Layout(tmp_path / "repro")
    rep = b.reproduce(1, [0], layout, out)
    assert rep["seeds"][0]["identical"]                    # same machine, same seed -> bit-identical training
    with pytest.raises(ValueError):
        b.reproduce(1, [0], layout, layout)                # never overwrite the reference results


def test_variance_ablation_curve_support_and_report(layout):
    v = b.run_variance(0, (5,), layout, SPEC)
    assert len(v["runs"]) == 2 and v["summary"]["CL"]["mae_sd"] >= 0
    a = b.run_ablation(0, ("plain", "full"), layout, SPEC)
    assert set(a["runs"]) == {"physics_only", "plain", "full"}
    assert a["runs"]["full"]["reflection_check"]["ok"] and not a["runs"]["plain"]["reflection_check"]["ok"]
    c = b.run_curve(0, (32, 64, 128), layout, SPEC)
    assert list(c["runs"]) == ["32", "64", "128"]
    s = b.run_support_study([0, 1], layout, n_airfoils=50, n_conditions=4)
    assert 0 <= s["runs"]["0"]["auc_nonconvergence"] <= 1
    summ = b.summarize_mains(layout.results)
    assert summ["n_replications"] == 2
    md = build_report(layout.root)
    for heading in ("Headline", "Per replication", "Optimisation variance", "Ablation", "Data efficiency"):
        assert heading in md


def test_auc_edge_cases():
    import math

    assert b._auc(np.array([0.1, 0.9]), np.array([False, True])) == 1.0
    assert b._auc(np.array([0.9, 0.1]), np.array([False, True])) == 0.0
    assert math.isnan(b._auc(np.array([0.1, 0.9]), np.array([False, False])))


def test_curve_rejects_a_broken_sobol_prefix(layout, monkeypatch):
    """Defence in depth: the data-efficiency curve relies on Sobol prefixes being balanced sub-designs."""
    import neuralnaca.domain as domain_mod

    real = domain_mod.sample_points
    monkeypatch.setattr(domain_mod, "sample_points", lambda n, kind, seed, *a: real(n, kind, seed + 1, *a))
    with pytest.raises(RuntimeError, match="Sobol prefix"):
        b.run_curve(0, (32,), layout, SPEC)
