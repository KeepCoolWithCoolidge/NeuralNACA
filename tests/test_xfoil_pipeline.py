"""The experiment code that needs a real XFOIL, exercised end to end at tiny scale (skipped without XFOIL)."""

import json

import numpy as np
import pytest

from neuralnaca import benchmark as b
from neuralnaca.cli import main
from neuralnaca.data import XfoilDataset
from neuralnaca.training import TrainConfig

pytestmark = [pytest.mark.xfoil, pytest.mark.neuralfoil, pytest.mark.slow]

SPEC = b.BenchmarkSpec(n_train=64, n_test=24, train=TrainConfig(ensemble=2, steps=40, width=16, depth=2))


@pytest.fixture(scope="module")
def layout(tmp_path_factory):
    lay = b.Layout(tmp_path_factory.mktemp("xbench"))
    b.run_main(0, lay, SPEC, workers=2, nf_sizes=("xxxlarge",))         # generates real XFOIL data
    return lay


def test_main_generates_data_with_provenance(layout):
    ds = XfoilDataset.load(layout.dataset("train", SPEC.n_train, 0, SPEC.xfoil))
    assert ds.meta["xfoil"]["build"] and ds.meta["design"]["kind"] == "sobol"
    assert ds.converged.mean() > 0.8


def test_protocol_study(layout):
    r = b.run_protocol_study(0, layout, SPEC, workers=2)
    assert set(r["truths"]) == {"cold/naca", "cold/naca-normalized", "cold/kulfan", "nf-sweep/naca",
                                "nf-sweep/kulfan"}
    assert r["n_common"] > 15
    pairs = r["xfoil_vs_xfoil"]
    # the frame transformation alone changes XFOIL's lift far less than the Kulfan shape fit
    assert pairs["cold/naca vs cold/naca-normalized"]["CL_MAE"] < pairs["cold/naca-normalized vs cold/kulfan"]["CL_MAE"]


def test_curve_beyond_the_shipped_training_set(layout):
    c = b.run_curve(0, (32, 64, 96), layout, SPEC, workers=2)
    assert list(c["runs"]) == ["32", "64", "96"] and c["runs"]["96"]["n_train_runs"] == 96
    big = XfoilDataset.load(layout.dataset("train", 96, 0, SPEC.xfoil))
    np.testing.assert_array_equal(big.points[:64], XfoilDataset.load(
        layout.dataset("train", 64, 0, SPEC.xfoil)).points)


def test_reproduce_level2(layout, tmp_path, monkeypatch):
    monkeypatch.setattr(b, "BenchmarkSpec", lambda: SPEC)
    rep = b.reproduce(2, [0], layout, b.Layout(tmp_path))
    row = rep["seeds"][0]
    assert row["data_agreement"]["CL_MedAE"] < 1e-6            # same XFOIL build -> identical labels
    assert row["identical"]


def test_polar_reference_and_speed(layout):
    path = b.run_polar_reference(layout, workers=2)
    z = np.load(path)
    assert z["xfoil"].shape == z["neuralfoil"].shape and np.isfinite(z["xfoil"][:, 0]).mean() > 0.8
    s = b.run_speed(layout, repeats=1)
    assert {"NeuralNACA", "NeuralFoil xxxlarge"} <= set(s["seconds_per_condition"])


def test_cli_generate_and_xfoil_check(tmp_path, capsys):
    out = tmp_path / "g.npz"
    assert main(["generate", "--n", "8", "--kind", "random", "--seed", "3", "--out", str(out), "--workers", "1"]) == 0
    assert XfoilDataset.load(out).meta["design"]["seed"] == 3
    assert main(["xfoil-check"]) == 0
    assert "PASS" in capsys.readouterr().out


def test_cli_bench_and_info(layout, capsys):
    assert main(["bench", "support", "--seeds", "0", "--root", str(layout.root)]) == 0
    assert json.loads((layout.results / "support.json").read_text())["seeds"] == [0]
    capsys.readouterr()
    assert main(["info"]) == 0
    assert json.loads(capsys.readouterr().out)["xfoil"].startswith(("xfoil-python", "XFOIL"))
