import json

import numpy as np
import pytest

from neuralnaca.data import (
    XfoilDataset,
    airfoil_keys,
    generate_dataset,
    sha256,
    split_train_test_overlap,
    verify_manifest,
)
from neuralnaca.domain import sample_points
from neuralnaca.xfoil import XfoilConfig


def make(n=10, seed=0):
    pts = sample_points(n, "random", seed)
    res = np.column_stack([pts[:, 4] * 0.1, np.full(n, 0.01), np.full(n, -0.05)])
    res[0] = np.nan
    return XfoilDataset(pts, res, {"note": "x"})


def test_dataset_round_trip(tmp_path):
    ds = make()
    ds.save(tmp_path / "d.npz")
    back = XfoilDataset.load(tmp_path / "d.npz")
    np.testing.assert_array_equal(back.points, ds.points)
    np.testing.assert_array_equal(back.results, ds.results)
    assert back.meta == {"note": "x"}
    assert len(back) == 10 and back.converged.sum() == 9
    assert back.targets().shape == (9, 3)
    np.testing.assert_allclose(back.targets()[:, 1], np.log(0.01))
    assert len(ds.subset(slice(0, 4))) == 4


def test_loading_files_without_metadata(tmp_path):
    ds = make()
    with open(tmp_path / "old.npz", "wb") as fh:
        np.savez(fh, points=ds.points, results=ds.results)
    assert XfoilDataset.load(tmp_path / "old.npz").meta == {}


def test_dataset_validation():
    with pytest.raises(ValueError):
        XfoilDataset(np.zeros((3, 4)), np.zeros((3, 3)))
    with pytest.raises(ValueError):
        XfoilDataset(np.zeros((3, 5)), np.zeros((2, 3)))


def test_nonpositive_drag_counts_as_unconverged():
    ds = make()
    ds.results[3, 1] = 0.0
    assert not ds.converged[3]


def test_overlap_detection():
    a, b = make(20, 0), make(20, 1)
    assert split_train_test_overlap(a, b) == 0
    assert split_train_test_overlap(a, a) == a.converged.sum()
    assert len(airfoil_keys(a.points)) == 20


def test_generate_dataset_uses_and_validates_the_cache(tmp_path, monkeypatch):
    import neuralnaca.data as data_mod

    calls = []

    def fake_run(points, cfg, workers, label):
        calls.append(len(points))
        return np.column_stack([points[:, 4] * 0.1, np.full(len(points), 0.01), np.zeros(len(points))])

    monkeypatch.setattr(data_mod, "run_xfoil", fake_run)
    path = tmp_path / "c.npz"
    ds = generate_dataset(16, "sobol", 0, XfoilConfig(), path=path)
    assert calls == [16] and ds.meta["design"]["kind"] == "sobol" and ds.meta["xfoil"]["panels"] == 279
    generate_dataset(16, "sobol", 0, XfoilConfig(), path=path)                       # cached
    assert calls == [16]
    generate_dataset(16, "sobol", 1, XfoilConfig(), path=path)                       # other design -> recompute
    generate_dataset(16, "sobol", 1, XfoilConfig(protocol="nf-sweep"), path=path)    # other protocol -> recompute
    generate_dataset(16, "sobol", 1, XfoilConfig(protocol="nf-sweep", n_crit=7.0), path=path)   # other n_crit
    generate_dataset(16, "sobol", 1, XfoilConfig(protocol="nf-sweep", n_crit=7.0), path=path)   # cached again
    assert calls == [16, 16, 16, 16]


def test_hashes_and_manifest(tmp_path):
    (tmp_path / "a.txt").write_bytes(b"abc")
    assert sha256(tmp_path / "a.txt") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"files": {"a.txt": sha256(tmp_path / "a.txt"),
                                                                   "missing.txt": "0" * 64}}), encoding="utf-8")
    assert verify_manifest(tmp_path / "MANIFEST.json") == {"a.txt": True, "missing.txt": False}
    (tmp_path / "a.txt").write_bytes(b"abd")
    assert verify_manifest(tmp_path / "MANIFEST.json")["a.txt"] is False
