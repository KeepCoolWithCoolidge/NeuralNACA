"""Integrity of everything shipped in the repository: data, models, results, figures inputs and documentation numbers.

These tests make the published claims executable: if a result file, a model or a number in the documentation
drifts from the others, the suite fails.
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from neuralnaca.benchmark import Layout, reproduce, summarize_mains
from neuralnaca.data import XfoilDataset, airfoil_keys, sha256, split_train_test_overlap, verify_manifest
from neuralnaca.domain import DEFAULT_DOMAIN, sample_points
from neuralnaca.evaluation import COLUMNS

pytestmark = pytest.mark.benchmark

ROOT = Path(__file__).resolve().parents[1]
LAYOUT = Layout(ROOT)
MAINS = sorted(LAYOUT.results.glob("main_s*.json")) if LAYOUT.results.exists() else []
SEEDS = [int(p.stem.split("_s")[1]) for p in MAINS if p.stem.count("_") == 1]


def main(seed):
    return json.loads((LAYOUT.results / f"main_s{seed}.json").read_text(encoding="utf-8"))


def test_manifest_checksums():
    status = verify_manifest(ROOT / "benchmarks" / "MANIFEST.json")
    assert status and all(status.values()), [k for k, ok in status.items() if not ok]


def test_at_least_five_replications():
    assert len(SEEDS) >= 5


@pytest.mark.parametrize("seed", SEEDS)
def test_datasets_follow_their_declared_design(seed):
    r = main(seed)
    for split, kind, s in (("train", "sobol", seed), ("test", "random", seed + 1000)):
        ds = XfoilDataset.load(LAYOUT.data / r["datasets"][split]["file"])
        assert ds.meta["design"]["kind"] == kind and ds.meta["design"]["seed"] == s
        # Re = 10**x goes through the platform's libm, which may differ in the last bit between operating systems
        # and library versions; every other column, and the design itself, must match exactly.
        want = sample_points(len(ds), kind, s)
        np.testing.assert_array_equal(ds.points[:, [0, 1, 2, 4]], want[:, [0, 1, 2, 4]])
        np.testing.assert_allclose(ds.points[:, 3], want[:, 3], rtol=1e-12, atol=0)
        assert DEFAULT_DOMAIN.contains(*ds.points.T).all()
        assert ds.converged.mean() > 0.9
        assert ds.meta["xfoil"]["panels"] == 279 and ds.meta["xfoil"]["n_crit"] == 9.0
        assert ds.meta["xfoil"]["protocol"] == "cold" and ds.meta["xfoil"]["geometry"] == "naca"


@pytest.mark.parametrize("seed", SEEDS)
def test_no_test_airfoil_was_seen_in_training(seed):
    r = main(seed)
    tr = XfoilDataset.load(LAYOUT.data / r["datasets"]["train"]["file"])
    te = XfoilDataset.load(LAYOUT.data / r["datasets"]["test"]["file"])
    assert split_train_test_overlap(tr, te) == 0 and r["leakage_test_airfoils_in_train"] == 0


@pytest.mark.parametrize("seed", SEEDS)
def test_result_records_are_consistent(seed):
    r = main(seed)
    assert sha256(LAYOUT.models / r["model"]["file"]) == r["model"]["sha256"]
    for split in ("train", "test"):
        assert sha256(LAYOUT.data / r["datasets"][split]["file"]) == r["datasets"][split]["sha256"]
    assert r["reflection_check"]["ok"]
    assert r["model"]["train_config"]["steps"] == 18000 and r["model"]["train_config"]["ensemble"] == 5
    assert r["datasets"]["train"]["n"] == 4096


@pytest.mark.parametrize("seed", SEEDS)
def test_central_claim_per_replication(seed):
    """NeuralNACA's MAE is lower than NeuralFoil-xxxlarge's for CL, ln CD and CM (95 % CI excludes 1)."""
    vs = main(seed)["comparison"]["vs"]["NeuralFoil xxxlarge"]
    for c in COLUMNS:
        assert vs["mae_ratio"][c][2] < 1.0, (seed, c, vs["mae_ratio"][c])


@pytest.mark.parametrize("seed", SEEDS)
def test_claim_holds_against_every_neuralfoil_size(seed):
    from neuralnaca.baselines import NEURALFOIL_SIZES

    vs = main(seed)["comparison"]["vs"]
    assert {f"NeuralFoil {z}" for z in NEURALFOIL_SIZES} <= set(vs)
    for name, v in vs.items():
        for c in COLUMNS:
            assert v["mae_ratio"][c][2] < 1.0, (seed, name, c)


def test_pooled_claim():
    s = summarize_mains(LAYOUT.results)
    for c in COLUMNS:
        assert s["pooled_mae_ratio"][c][2] < 1.0


def test_shipped_package_weights_are_the_seed0_benchmark_model():
    pkg = ROOT / "src" / "neuralnaca" / "weights" / "naca4_k5_n4096_s0.npz"
    assert sha256(pkg) == main(0)["model"]["sha256"]


def test_shipped_support_model_is_built_from_the_seed0_training_set():
    from neuralnaca.support import SupportModel

    shipped = SupportModel.load(ROOT / "src" / "neuralnaca" / "weights" / "naca4_k5_n4096_s0.support.npz")
    rebuilt = SupportModel.from_dataset(XfoilDataset.load(LAYOUT.data / main(0)["datasets"]["train"]["file"]))
    np.testing.assert_array_equal(shipped.points, rebuilt.points)
    np.testing.assert_array_equal(shipped.converged, rebuilt.converged)
    assert shipped.cd_max == rebuilt.cd_max


def test_documentation_figure_sections_are_unseen():
    z = np.load(LAYOUT.results / "polars_reference.npz")
    seen = set()
    for p in LAYOUT.data.glob("train_*.npz"):
        seen |= airfoil_keys(XfoilDataset.load(p).points)
    assert not (airfoil_keys(z["points"]) & seen)


@pytest.mark.neuralfoil
def test_reproduce_level0_all_replications():
    rep = reproduce(0, SEEDS, LAYOUT)
    for seed, row in rep["seeds"].items():
        assert row["files_ok"], seed
        assert row["metrics_within_tolerance"], (seed, row["relative_deviation"])
        assert row["claim_holds"], seed
    assert rep["passed"]


def test_generated_tables_are_up_to_date():
    from neuralnaca.report import build_headline, build_report

    assert (ROOT / "docs" / "results" / "tables.md").read_text(encoding="utf-8") == build_report(ROOT), \
        "run `neuralnaca report` to regenerate docs/results/tables.md"
    assert (ROOT / "docs" / "results" / "headline.md").read_text(encoding="utf-8") == build_headline(ROOT)


def test_headline_numbers_in_readme_match_the_results():
    """Every pooled MAE ratio and interval quoted in the README is the one computed from the shipped results."""
    s = summarize_mains(LAYOUT.results)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for c in COLUMNS:
        r, lo, hi = s["pooled_mae_ratio"][c]
        assert f"{r:.2f} [{lo:.2f}, {hi:.2f}]" in readme, c
    assert f"{len(SEEDS)} independent replications" in readme
    n_runs = main(0)["datasets"]["train"]["n"]
    assert f"{n_runs:,}" in readme


def test_no_unexplained_numbers_in_docs():
    """Ratios written in the prose pages (x.xx [y.yy, z.zz]) must appear in the generated tables."""
    tables = (ROOT / "docs" / "results" / "tables.md").read_text(encoding="utf-8")
    pattern = re.compile(r"\d\.\d\d \[\d\.\d\d, \d\.\d\d\]")
    for page in [*(ROOT / "docs").rglob("*.md"), ROOT / "README.md"]:
        if page.name == "tables.md":
            continue
        for hit in pattern.findall(page.read_text(encoding="utf-8")):
            assert hit in tables, (page.name, hit)


def test_figures_regenerate(tmp_path):
    pytest.importorskip("matplotlib")
    from neuralnaca.plotting import make_figures

    paths = make_figures(ROOT, tmp_path)
    names = {p.name for p in paths}
    assert {"polars_re1e06.png", "polars_re2e05.png", "data_efficiency.png", "ablation.png", "error_cdf.png",
            "support_map.png"} <= names
    assert all(p.stat().st_size > 20_000 for p in paths)
