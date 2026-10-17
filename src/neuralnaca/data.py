"""XFOIL data sets: a small, self-describing container plus generation and integrity helpers."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import platform
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .domain import DEFAULT_DOMAIN, Domain, sample_points
from .xfoil import XfoilConfig, run_xfoil, xfoil_version

__all__ = [
    "XfoilDataset",
    "airfoil_keys",
    "generate_dataset",
    "pack_dataset",
    "sha256",
    "split_train_test_overlap",
    "verify_manifest",
]

log = logging.getLogger(__name__)


def airfoil_keys(points: np.ndarray) -> set[tuple[float, float, float]]:
    """Identity of the airfoils in a point set (``m, p, t`` rounded to 1e-9)."""
    return {tuple(np.round(r[:3], 9)) for r in np.asarray(points)}


@dataclass
class XfoilDataset:
    """Flow conditions and the XFOIL answers for them.

    Attributes
    ----------
    points
        ``(n, 5)`` ``[m, p, t, Re, alpha]``.
    results
        ``(n, 3)`` ``[CL, CD, CM]``; NaN rows where XFOIL did not converge. Non-converged points are *kept* so that
        the data set documents the full experimental design and the convergence rate.
    meta
        Provenance: sampling design, XFOIL configuration and build, creation time, software versions.
    """

    points: np.ndarray
    results: np.ndarray
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=float)
        self.results = np.asarray(self.results, dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] != 5:
            raise ValueError("points must have shape (n, 5)")
        if self.results.shape != (len(self.points), 3):
            raise ValueError("results must have shape (n, 3)")

    def __len__(self) -> int:
        return len(self.points)

    @property
    def converged(self) -> np.ndarray:
        """Boolean mask of points where XFOIL converged."""
        return np.isfinite(self.results).all(1) & (self.results[:, 1] > 0)

    def targets(self) -> np.ndarray:
        """``[CL, ln CD, CM]`` of the converged points."""
        r = self.results[self.converged]
        return np.stack([r[:, 0], np.log(r[:, 1]), r[:, 2]], 1)

    def subset(self, idx) -> XfoilDataset:
        return XfoilDataset(self.points[idx], self.results[idx], dict(self.meta))

    def save(self, path: str | Path) -> None:
        """Write a ``.npz`` file (arrays + JSON metadata; no pickle)."""
        with open(path, "wb") as fh:
            np.savez(fh, points=self.points, results=self.results,
                     meta=np.array(json.dumps(self.meta, sort_keys=True, default=str)))

    @classmethod
    def load(cls, path: str | Path) -> XfoilDataset:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
            return cls(z["points"], z["results"], meta)


def generate_dataset(n: int, kind: str, seed: int, cfg: XfoilConfig, workers: int | None = None,
                     domain: Domain = DEFAULT_DOMAIN, path: str | Path | None = None, label: str = "data",
                     points: np.ndarray | None = None) -> XfoilDataset:
    """Sample a design and run XFOIL on it (or reuse ``path`` if it already holds exactly these points).

    Parameters
    ----------
    n, kind, seed, domain
        Passed to :func:`neuralnaca.domain.sample_points` (ignored if ``points`` is given).
    cfg
        XFOIL settings.
    workers
        Parallel XFOIL processes.
    path
        Optional cache file. If it exists and contains the same points *and* the same XFOIL configuration, it is
        loaded instead of re-running XFOIL.
    points
        Explicit points (overrides the sampled design).
    """
    pts = sample_points(n, kind, seed, domain) if points is None else np.asarray(points, dtype=float)
    if path is not None and Path(path).exists():
        ds = XfoilDataset.load(path)
        rec = ds.meta.get("xfoil", {})
        defaults = {"protocol": "cold", "geometry": "naca", "panels": 279, "n_crit": 9.0, "max_iter": 100}
        same_cfg = all(rec.get(k, d) == getattr(cfg, k) for k, d in defaults.items())
        if ds.points.shape == pts.shape and np.allclose(ds.points, pts, rtol=1e-12, atol=0) and same_cfg:
            log.info("[%s] using cached XFOIL results %s", label, path)
            return ds
        log.info("[%s] cache %s does not match the requested design/configuration; recomputing", label, path)
    log.info("[%s] running XFOIL on %d points (backend=%s, protocol=%s, geometry=%s)", label, len(pts), cfg.backend,
             cfg.protocol, cfg.geometry)
    t0 = _dt.datetime.now(_dt.timezone.utc)
    res = run_xfoil(pts, cfg, workers, label)
    meta = {
        "design": {"n": len(pts), "kind": kind if points is None else "explicit", "seed": seed,
                   "domain": domain.to_dict()},
        "xfoil": cfg.to_dict() | {"build": xfoil_version(cfg)},
        "created_utc": t0.isoformat(timespec="seconds"),
        "runtime_s": round((_dt.datetime.now(_dt.timezone.utc) - t0).total_seconds(), 1),
        "platform": platform.platform(),
    }
    ds = XfoilDataset(pts, res, meta)
    ok = ds.converged
    log.info("[%s] %d/%d converged (%.1f%%)", label, ok.sum(), len(ds), 100 * ok.mean())
    if path is not None:
        ds.save(path)
    return ds


def pack_dataset(ds: XfoilDataset, with_targets: bool = True) -> dict:
    """Converged points of a data set as model inputs (panel constants computed) - the input of
    :func:`neuralnaca.training.train_ensemble` and :func:`neuralnaca.model.predict_tensors`."""
    from .model import to_tensors
    from .panel import airfoil_constants

    ok = ds.converged
    return to_tensors(ds.points[ok], airfoil_constants(ds.points[ok, :3]), ds.results[ok] if with_targets else None)


def split_train_test_overlap(train: XfoilDataset, test: XfoilDataset) -> int:
    """Number of *converged test points* whose airfoil also occurs in the converged training set (should be 0)."""
    keys = airfoil_keys(train.points[train.converged])
    return sum(tuple(np.round(r[:3], 9)) in keys for r in test.points[test.converged])


_TEXT_SUFFIXES = frozenset({".json", ".txt", ".md"})


def sha256(path: str | Path) -> str:
    """SHA-256 of a file. Text files (.json, .txt, .md) are hashed with CRLF read as LF, so that a Windows checkout
    with ``core.autocrlf`` yields the same digest as the repository; binary files are hashed byte for byte."""
    path = Path(path)
    if path.suffix.lower() in _TEXT_SUFFIXES:
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_manifest(manifest: str | Path) -> dict[str, bool]:
    """Check the SHA-256 of every file listed in a manifest (paths relative to the manifest's directory)."""
    manifest = Path(manifest)
    entries = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    return {rel: (manifest.parent / rel).exists() and sha256(manifest.parent / rel) == digest
            for rel, digest in entries.items()}
