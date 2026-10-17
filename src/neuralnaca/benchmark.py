"""The benchmark: every reported number is produced by the functions in this module.

Layout of a benchmark root (the repository root by default)::

    data/xfoil/                 XFOIL data sets (train_<n>_s<seed>_<tag>.npz, test_<n>_s<seed>_<tag>.npz)
    benchmarks/models/          trained weights (<variant>_k<K>_n<n>_s<seed>[_ts<train seed>].npz)
    benchmarks/results/         JSON results (+ per-point predictions as .npz)
    benchmarks/MANIFEST.json    SHA-256 of every shipped data / model / result file

Experiments
-----------
``main``      one replication: fresh Sobol training set + i.i.d. test set (seed), train, evaluate vs NeuralFoil
``variance``  retrain on the *same* data with different training seeds (spread due to optimisation alone)
``ablation``  remove one physics ingredient at a time (same data, same recipe)
``curve``     error vs number of XFOIL training runs (Sobol prefixes)
``protocol``  sensitivity of the comparison to the definition of the XFOIL ground truth
``reproduce`` re-run / re-check the shipped results at three levels of depth
"""

from __future__ import annotations

import json
import logging
import math
import platform
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ._version import __version__
from .baselines import NEURALFOIL_SIZES, NF_TRAIN_RUNS, neuralfoil_predict, neuralfoil_version
from .data import XfoilDataset, generate_dataset, pack_dataset, sha256, split_train_test_overlap
from .domain import DEFAULT_DOMAIN
from .evaluation import COLUMNS, bootstrap_ratio, compare, error_metrics
from .model import FoilNet, check_reflection, load_model, predict_tensors, save_model, to_tensors
from .panel import airfoil_constants
from .training import TrainConfig, train_ensemble
from .xfoil import XfoilConfig, detect_xfoil

__all__ = [
    "BenchmarkSpec",
    "Layout",
    "environment_info",
    "reproduce",
    "run_ablation",
    "run_curve",
    "run_main",
    "run_polar_reference",
    "run_protocol_study",
    "run_speed",
    "run_support_study",
    "run_variance",
]

log = logging.getLogger(__name__)

PRIMARY_NF = "xxxlarge"


@dataclass(frozen=True)
class BenchmarkSpec:
    """Size of one replication. The defaults are the reported configuration."""

    n_train: int = 4096
    n_test: int = 3000
    train: TrainConfig = TrainConfig()
    xfoil: XfoilConfig = XfoilConfig()


@dataclass(frozen=True)
class Layout:
    """File layout of a benchmark root directory."""

    root: Path = Path(".")

    @property
    def data(self) -> Path:
        return Path(self.root) / "data" / "xfoil"

    @property
    def models(self) -> Path:
        return Path(self.root) / "benchmarks" / "models"

    @property
    def results(self) -> Path:
        return Path(self.root) / "benchmarks" / "results"

    def mkdirs(self) -> None:
        for d in (self.data, self.models, self.results):
            d.mkdir(parents=True, exist_ok=True)

    def dataset(self, split: str, n: int, seed: int, xcfg: XfoilConfig) -> Path:
        return self.data / f"{split}_{n}_s{seed}_{xcfg.tag()}.npz"

    def model(self, variant: str, K: int, n: int, seed: int, train_seed: int | None = None,
              residuals: bool = False) -> Path:
        ts = "" if train_seed is None or train_seed == seed else f"_ts{train_seed}"
        res = "_res" if residuals else ""
        return self.models / f"{variant}{res}_k{K}_n{n}_s{seed}{ts}.npz"


def environment_info() -> dict:
    """Versions of everything that can influence the numbers."""
    import scipy
    import torch

    info = {"neuralnaca": __version__, "python": sys.version.split()[0], "platform": platform.platform(),
            "machine": platform.machine(), "numpy": np.__version__, "scipy": scipy.__version__,
            "torch": torch.__version__, "torch_threads": torch.get_num_threads(), "neuralfoil": neuralfoil_version()}
    try:
        from importlib.metadata import version

        info["aerosandbox"] = version("aerosandbox")
    except Exception:
        info["aerosandbox"] = None
    return info


def _dataset_record(path: Path, ds: XfoilDataset) -> dict:
    return {"file": path.name, "sha256": sha256(path), "n": len(ds), "converged": int(ds.converged.sum())}


_packed = pack_dataset


def _predict(net: FoilNet, ds: XfoilDataset) -> np.ndarray:
    return predict_tensors(net, _packed(ds, with_targets=False)).mean(0)


def _load_or_train(path: Path, data_tr: dict, cfg: TrainConfig, meta: dict, retrain: bool, tag: str,
                   col: dict | None = None) -> tuple[FoilNet, dict, bool]:
    if path.exists() and not retrain:
        net, m = load_model(path)
        if m.get("train_config") == cfg.to_dict() and m.get("train_data_sha256") == meta.get("train_data_sha256"):
            log.info("[%s] loaded matching model %s", tag, path)
            return net, m, False
        log.info("[%s] %s was trained differently; retraining", tag, path)
    t0 = time.time()
    net = train_ensemble(data_tr, cfg, col=col, tag=tag)
    meta = meta | {"train_config": cfg.to_dict(), "train_minutes": round((time.time() - t0) / 60, 2),
                   "environment": environment_info()}
    save_model(net, path, meta)
    return net, meta, True


def _datasets(spec: BenchmarkSpec, seed: int, layout: Layout, workers: int | None,
              xcfg: XfoilConfig | None = None) -> tuple[XfoilDataset, XfoilDataset, Path, Path]:
    layout.mkdirs()
    xcfg = xcfg or spec.xfoil
    ptr = layout.dataset("train", spec.n_train, seed, xcfg)
    pte = layout.dataset("test", spec.n_test, seed, xcfg)
    tr = generate_dataset(spec.n_train, "sobol", seed, xcfg, workers, path=ptr, label=f"train s{seed}")
    te = generate_dataset(spec.n_test, "random", seed + 1000, xcfg, workers, path=pte, label=f"test s{seed}")
    return tr, te, ptr, pte


def _nf_predictions(points: np.ndarray, sizes: Sequence[str]) -> dict[str, np.ndarray]:
    out = {}
    for s in sizes:
        t0 = time.time()
        out[f"NeuralFoil {s}"] = neuralfoil_predict(points, s)
        log.info("NeuralFoil %s evaluated in %.0f s", s, time.time() - t0)
    return out


def _write(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=False, default=_json_default) + "\n", encoding="utf-8")


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


# ------------------------------------------------------------------------------------------------ main replication
def run_main(seed: int, layout: Layout = Layout(), spec: BenchmarkSpec = BenchmarkSpec(), workers: int | None = None,
             nf_sizes: Sequence[str] = NEURALFOIL_SIZES, retrain: bool = False, xcfg: XfoilConfig | None = None,
             write: bool = True) -> dict:
    """One full replication of the benchmark (data -> training -> evaluation against XFOIL and NeuralFoil).

    XFOIL is only needed if the data sets are not already present in ``layout.data``.
    """
    if xcfg is None and not (layout.dataset("train", spec.n_train, seed, spec.xfoil).exists()
                             and layout.dataset("test", spec.n_test, seed, spec.xfoil).exists()):
        xcfg = detect_xfoil(panels=spec.xfoil.panels)
    xcfg = xcfg or spec.xfoil
    tr, te, ptr, pte = _datasets(spec, seed, layout, workers, xcfg)
    leak = split_train_test_overlap(tr, te)
    if leak:
        raise RuntimeError(f"{leak} test points share an airfoil with the training set")
    cfg = replace(spec.train, seed=seed)
    mpath = layout.model(cfg.variant, cfg.ensemble, spec.n_train, seed)
    meta = {"train_data": ptr.name, "train_data_sha256": sha256(ptr), "n_train_runs": spec.n_train,
            "n_train_converged": int(tr.converged.sum()), "domain": DEFAULT_DOMAIN.to_dict()}
    net, meta, _ = _load_or_train(mpath, _packed(tr), cfg, meta, retrain, f"main s{seed}")
    refl = check_reflection(net)
    pred = _predict(net, te)
    ok = te.converged
    y = te.targets()
    nf = _nf_predictions(te.points[ok], nf_sizes) if nf_sizes and neuralfoil_version() else {}
    comp = compare(pred, nf, y, te.points[ok])
    res = {
        "kind": "main", "seed": seed, "neuralnaca": __version__,
        "datasets": {"train": _dataset_record(ptr, tr), "test": _dataset_record(pte, te)},
        "leakage_test_airfoils_in_train": leak,
        "model": {"file": mpath.name, "sha256": sha256(mpath), "train_config": cfg.to_dict(),
                  "train_minutes": meta.get("train_minutes")},
        "reflection_check": refl,
        "neuralfoil": {"version": neuralfoil_version(), "sizes": list(nf_sizes) if nf else []},
        "training_data_fraction_of_neuralfoil": spec.n_train / NF_TRAIN_RUNS,
        "comparison": comp,
        "environment": environment_info(),
    }
    if write:
        _write(res, layout.results / f"main_s{seed}.json")
        arrays = {"points": te.points[ok], "y": y, "model": pred}
        arrays |= {f"nf_{k.split()[-1]}": v for k, v in nf.items() if k.endswith(PRIMARY_NF)}
        with open(layout.results / f"main_s{seed}_pred.npz", "wb") as fh:
            np.savez(fh, **arrays)
    return res


# ------------------------------------------------------------------------------------------------ training variance
def run_variance(seed: int = 0, train_seeds: Sequence[int] = (101, 102, 103, 104), layout: Layout = Layout(),
                 spec: BenchmarkSpec = BenchmarkSpec(), retrain: bool = False) -> dict:
    """Retrain on the same data with different training seeds; quantifies optimisation-only variability."""
    tr, te, ptr, _ = _datasets(spec, seed, layout, None)
    data_tr, y = _packed(tr), te.targets()
    ok = te.converged
    ref = np.load(layout.results / f"main_s{seed}_pred.npz")
    e_nf = np.abs(ref[f"nf_{PRIMARY_NF}"] - y)
    rows: list[dict] = []
    for ts in [seed, *train_seeds]:
        cfg = replace(spec.train, seed=ts)
        mpath = layout.model(cfg.variant, cfg.ensemble, spec.n_train, seed, train_seed=ts)
        meta = {"train_data": ptr.name, "train_data_sha256": sha256(ptr)}
        net, _, _ = _load_or_train(mpath, data_tr, cfg, meta, retrain, f"variance ts{ts}")
        pred = _predict(net, te)
        assert np.array_equal(te.points[ok], ref["points"])
        e = np.abs(pred - y)
        rows.append({"train_seed": ts, "metrics": error_metrics(pred, y),
                     "mae_ratio_vs_nf": dict(zip(COLUMNS, map(float, e.mean(0) / e_nf.mean(0))))})
    summ = {}
    for c in COLUMNS:
        v = np.array([r["metrics"][f"{c}_MAE"] for r in rows])
        q = np.array([r["mae_ratio_vs_nf"][c] for r in rows])
        summ[c] = {"mae_mean": float(v.mean()), "mae_sd": float(v.std(ddof=1)), "ratio_mean": float(q.mean()),
                   "ratio_sd": float(q.std(ddof=1)), "ratio_max": float(q.max())}
    res = {"kind": "variance", "seed": seed, "train_seeds": [seed, *train_seeds], "runs": rows, "summary": summ,
           "environment": environment_info()}
    _write(res, layout.results / f"variance_s{seed}.json")
    return res


# ------------------------------------------------------------------------------------------------ ablation
ABLATION_VARIANTS = ("plain", "no_panel", "no_symmetry", "no_drag_prior", "full")


def run_ablation(seed: int, variants: Sequence[str] = ABLATION_VARIANTS, layout: Layout = Layout(),
                 spec: BenchmarkSpec = BenchmarkSpec(), residuals: bool = False, retrain: bool = False) -> dict:
    """Train each model variant on the same data with the same recipe and evaluate on the same test points.

    With ``residuals=True`` the full model is additionally trained with the autograd physics residuals.
    """
    tr, te, ptr, _ = _datasets(spec, seed, layout, None)
    data_tr, y = _packed(tr), te.targets()
    ok = te.converged
    ref = np.load(layout.results / f"main_s{seed}_pred.npz")
    assert np.array_equal(te.points[ok], ref["points"])
    e_nf = np.abs(ref[f"nf_{PRIMARY_NF}"] - y)
    out_path = layout.results / f"ablation_s{seed}.json"
    res: dict = (json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists()
                 else {"kind": "ablation", "seed": seed, "runs": {}})
    # physics alone (panel-method CL/CM, flat-plate drag prior), no learning: what the network adds on top of it
    from .model import make_inputs

    d = _packed(te, with_targets=False)
    _, _, base = make_inputs(d["m"], d["p"], d["t"], d["lnre"], d["alpha"], d["const"])
    pb = base.numpy().astype(float)
    eb_ = np.abs(pb - y)
    r, lo, hi = bootstrap_ratio(eb_, e_nf)
    res["runs"]["physics_only"] = {
        "variant": "physics_only", "residuals": False, "metrics": error_metrics(pb, y),
        "mae_ratio_vs_nf": {c: [float(r[j]), float(lo[j]), float(hi[j])] for j, c in enumerate(COLUMNS)},
        "reflection_check": None, "train_minutes": 0.0,
    }
    jobs = [(v, False) for v in variants] + ([("full", True)] if residuals else [])
    col = None
    for variant, use_res in jobs:
        name = variant + ("+residuals" if use_res else "")
        cfg = replace(spec.train, seed=seed, variant=variant, residuals=use_res)
        mpath = layout.model(variant, cfg.ensemble, spec.n_train, seed, residuals=use_res)
        if use_res and col is None:
            from .domain import sample_points

            cpts = sample_points(4096, "sobol", seed + 7)
            col = to_tensors(cpts, airfoil_constants(cpts[:, :3]))
        meta = {"train_data": ptr.name, "train_data_sha256": sha256(ptr)}
        net, m, _ = _load_or_train(mpath, data_tr, cfg, meta, retrain, f"ablation {name} s{seed}", col=col)
        pred = _predict(net, te)
        e = np.abs(pred - y)
        r, lo, hi = bootstrap_ratio(e, e_nf)
        res["runs"][name] = {
            "variant": variant, "residuals": use_res, "metrics": error_metrics(pred, y),
            "mae_ratio_vs_nf": {c: [float(r[j]), float(lo[j]), float(hi[j])] for j, c in enumerate(COLUMNS)},
            "reflection_check": check_reflection(net), "train_minutes": m.get("train_minutes"),
        }
        res["environment"] = environment_info()
        _write(res, out_path)
    return res


# ------------------------------------------------------------------------------------------------ data curve
def run_curve(seed: int, sizes: Sequence[int] = (512, 1024, 2048, 4096), layout: Layout = Layout(),
              spec: BenchmarkSpec = BenchmarkSpec(), workers: int | None = None, retrain: bool = False) -> dict:
    """Error versus number of XFOIL training runs, using prefixes of the Sobol training sequence.

    Sizes above ``spec.n_train`` extend the shipped training set (needs XFOIL for the extra points only).
    """
    tr, te, ptr, _ = _datasets(spec, seed, layout, workers)
    y = te.targets()
    ok = te.converged
    ref = np.load(layout.results / f"main_s{seed}_pred.npz")
    assert np.array_equal(te.points[ok], ref["points"])
    e_nf = np.abs(ref[f"nf_{PRIMARY_NF}"] - y)
    out_path = layout.results / f"curve_s{seed}.json"
    res: dict = (json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists()
                 else {"kind": "curve", "seed": seed, "runs": {}})
    from .domain import sample_points

    for n in sizes:
        if n <= spec.n_train:
            sub, src = tr.subset(slice(0, n)), ptr
            if not np.array_equal(sub.points, sample_points(n, "sobol", seed)):
                raise RuntimeError("Sobol prefix property violated")
        else:
            big = layout.dataset("train", n, seed, spec.xfoil)
            pts = sample_points(n, "sobol", seed)
            if not np.array_equal(pts[: spec.n_train], tr.points):
                raise RuntimeError("Sobol prefix property violated")
            if big.exists():
                sub = XfoilDataset.load(big)
            else:
                xcfg = detect_xfoil(panels=spec.xfoil.panels)
                extra = generate_dataset(0, "sobol", seed, xcfg, workers, points=pts[spec.n_train:],
                                         label=f"train extension s{seed}")
                sub = XfoilDataset(pts, np.concatenate([tr.results, extra.results]),
                                   tr.meta | {"design": tr.meta.get("design", {}) | {"n": n}})
                sub.save(big)
            src = big
        cfg = replace(spec.train, seed=seed)
        mpath = layout.model(cfg.variant, cfg.ensemble, n, seed)
        digest = sha256(src) if n >= spec.n_train else f"{sha256(ptr)}[:{n}]"
        meta = {"train_data": src.name, "train_data_sha256": digest, "n_train_runs": n}
        net, _, _ = _load_or_train(mpath, _packed(sub), cfg, meta, retrain, f"curve n={n} s{seed}")
        pred = _predict(net, te)
        e = np.abs(pred - y)
        r, lo, hi = bootstrap_ratio(e, e_nf)
        res["runs"][str(n)] = {"n_train_runs": n, "n_converged": int(sub.converged.sum()),
                               "metrics": error_metrics(pred, y),
                               "mae_ratio_vs_nf": {c: [float(r[j]), float(lo[j]), float(hi[j])]
                                                   for j, c in enumerate(COLUMNS)},
                               "fraction_of_neuralfoil_data": n / NF_TRAIN_RUNS}
        res["environment"] = environment_info()
        _write(res, out_path)
    return res


# ------------------------------------------------------------------------------------------------ protocol study
def protocol_truths(base: XfoilConfig) -> dict[str, XfoilConfig]:
    """The ground-truth definitions compared in the protocol study (``cold/naca-normalized`` is a control that
    isolates NeuralFoil's frame transformation from its Kulfan shape fit)."""
    return {
        "cold/naca": base,
        "cold/naca-normalized": replace(base, geometry="naca-normalized"),
        "cold/kulfan": replace(base, geometry="kulfan"),
        "nf-sweep/naca": base.neuralfoil_protocol(geometry="naca"),
        "nf-sweep/kulfan": base.neuralfoil_protocol(geometry="kulfan"),
    }


def run_protocol_study(seed: int = 0, layout: Layout = Layout(), spec: BenchmarkSpec = BenchmarkSpec(),
                       workers: int | None = None) -> dict:
    """How much does the conclusion depend on how the XFOIL ground truth is defined?

    The test points of replication ``seed`` are re-solved with XFOIL under four definitions (cold start vs.
    NeuralFoil's warm-started sweep protocol; exact NACA geometry vs. the Kulfan fit NeuralFoil analyses). The
    shipped model (trained on cold/naca data) and NeuralFoil are scored against each, on the points that converged
    under *all* definitions. XFOIL-vs-XFOIL differences between definitions are reported as well: they bound how
    precisely any surrogate can be said to "match XFOIL".
    """
    ref = json.loads((layout.results / f"main_s{seed}.json").read_text(encoding="utf-8"))
    te0 = XfoilDataset.load(layout.data / ref["datasets"]["test"]["file"])
    base = spec.xfoil
    truths = {}
    for name, xc in protocol_truths(base).items():
        if name == "cold/naca":
            truths[name] = te0
            continue
        path = layout.dataset("test", spec.n_test, seed, xc)
        if path.exists():
            truths[name] = XfoilDataset.load(path)
        else:
            run_cfg = replace(detect_xfoil(panels=base.panels), protocol=xc.protocol, geometry=xc.geometry,
                              max_iter=xc.max_iter)
            truths[name] = generate_dataset(spec.n_test, "random", seed + 1000, run_cfg, workers, path=path,
                                            label=f"truth {name}")
        assert np.array_equal(truths[name].points, te0.points)
    common = np.stack([d.converged for d in truths.values()]).all(axis=0)
    pts = te0.points[common]
    net, _ = load_model(layout.models / ref["model"]["file"])
    pred = predict_tensors(net, to_tensors(pts, airfoil_constants(pts[:, :3]))).mean(0)
    nf = neuralfoil_predict(pts, PRIMARY_NF)
    res: dict = {"kind": "protocol", "seed": seed, "n_common": int(common.sum()),
           "converged": {k: int(d.converged.sum()) for k, d in truths.items()}, "truths": {}, "xfoil_vs_xfoil": {},
           "datasets": {k: d.meta.get("xfoil", {}) for k, d in truths.items()}}
    ys = {}
    for name, d in truths.items():
        r = d.results[common]
        y = np.stack([r[:, 0], np.log(r[:, 1]), r[:, 2]], 1)
        ys[name] = y
        ea, eb = np.abs(pred - y), np.abs(nf - y)
        q, lo, hi = bootstrap_ratio(ea, eb)
        res["truths"][name] = {"model": error_metrics(pred, y), "neuralfoil": error_metrics(nf, y),
                               "mae_ratio": {c: [float(q[j]), float(lo[j]), float(hi[j])]
                                             for j, c in enumerate(COLUMNS)}}
    names = list(truths)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            e = np.abs(ys[a] - ys[b])
            res["xfoil_vs_xfoil"][f"{a} vs {b}"] = {
                **{f"{c}_MAE": float(e[:, j].mean()) for j, c in enumerate(COLUMNS)},
                **{f"{c}_MedAE": float(np.median(e[:, j])) for j, c in enumerate(COLUMNS)},
                "frac_CL_diff_gt_0.01": float((e[:, 0] > 0.01).mean())}
    res["environment"] = environment_info()
    _write(res, layout.results / f"protocol_s{seed}.json")
    return res


# ------------------------------------------------------------------------------------------------ reproduce
def _claims_hold(comp: dict) -> bool:
    vs = comp["vs"].get(f"NeuralFoil {PRIMARY_NF}")
    return bool(vs) and all(vs["mae_ratio"][c][2] < 1.0 for c in COLUMNS)


def reproduce(level: int = 0, seeds: Sequence[int] | None = None, layout: Layout = Layout(),
              out: Layout | None = None, workers: int | None = None, rel_tol: float | None = None) -> dict:
    """Re-derive the shipped results and compare.

    Levels
    ------
    0
        *Verify*: shipped weights + shipped test data -> metrics (seconds). Must agree with the shipped results to
        floating-point precision (``rel_tol`` default 1e-4).
    1
        *Retrain*: shipped training data -> new weights -> metrics (~7 min per seed on 2 CPU cores). Identical on
        the reference platform; elsewhere the model metrics must fall inside the band set by the training-seed
        variance study (``rel_tol`` default: 4 standard deviations of the MAE, at least 3 %).
    2
        *Regenerate*: your XFOIL -> new data sets -> retrain -> metrics (~20 min per seed). Data agreement with the
        shipped sets is reported; the claim check (CI upper bound of the MAE ratio < 1 for all three outputs) is
        the pass criterion.

    In every level the central claim is re-checked: NeuralNACA's MAE is lower than NeuralFoil-xxxlarge's for
    ``CL``, ``ln CD`` and ``CM`` with the 95 % paired-bootstrap interval excluding 1.
    """
    if level not in (0, 1, 2):
        raise ValueError("level must be 0, 1 or 2")
    seeds = list(seeds) if seeds is not None else sorted(
        int(p.stem.split("_s")[1]) for p in layout.results.glob("main_s*.json") if p.stem.count("_") == 1)
    if not seeds:
        raise FileNotFoundError(f"no reference results in {layout.results}")
    var_path = layout.results / "variance_s0.json"
    var = json.loads(var_path.read_text(encoding="utf-8"))["summary"] if var_path.exists() else None
    report: dict = {"level": level, "seeds": {}, "environment": environment_info()}
    for seed in seeds:
        ref = json.loads((layout.results / f"main_s{seed}.json").read_text(encoding="utf-8"))
        rm = ref["comparison"]["metrics"]["model"]
        row: dict = {}
        if level == 0:
            te_path = layout.data / ref["datasets"]["test"]["file"]
            m_path = layout.models / ref["model"]["file"]
            row["files_ok"] = (sha256(te_path) == ref["datasets"]["test"]["sha256"]
                               and sha256(m_path) == ref["model"]["sha256"])
            te = XfoilDataset.load(te_path)
            net, _ = load_model(m_path)
            pred, y = _predict(net, te), te.targets()
            nf = {}
            if neuralfoil_version():
                nf = {f"NeuralFoil {PRIMARY_NF}": neuralfoil_predict(te.points[te.converged], PRIMARY_NF)}
            comp = compare(pred, nf, y, te.points[te.converged], n_boot=2000)
            tol = rel_tol or 1e-4
        else:
            target = out or Layout(Path(layout.root) / "reproduction")
            if Path(target.results).resolve() == Path(layout.results).resolve():
                raise ValueError("levels 1 and 2 must write to a different directory than the reference results")
            spec = BenchmarkSpec()
            if level == 1:
                target.mkdirs()
                for split in ("train", "test"):
                    src = layout.data / ref["datasets"][split]["file"]
                    dst = target.data / src.name
                    if Path(src).resolve() != Path(dst).resolve():
                        dst.write_bytes(src.read_bytes())
                xcfg = spec.xfoil
            else:
                xcfg = detect_xfoil(panels=spec.xfoil.panels)
            r = run_main(seed, target, spec, workers, nf_sizes=(PRIMARY_NF,), retrain=True, xcfg=xcfg, write=True)
            comp = r["comparison"]
            if level == 2:
                mine = XfoilDataset.load(target.data / r["datasets"]["test"]["file"])
                theirs = XfoilDataset.load(layout.data / ref["datasets"]["test"]["file"])
                both = mine.converged & theirs.converged
                d = np.abs(mine.results[both] - theirs.results[both])
                row["data_agreement"] = {
                    "converged_mine": int(mine.converged.sum()), "converged_shipped": int(theirs.converged.sum()),
                    "both": int(both.sum()), "CL_MedAE": float(np.median(d[:, 0])),
                    "frac_CL_diff_gt_0.01": float((d[:, 0] > 0.01).mean())}
            if rel_tol is not None:
                tol = rel_tol
            elif var is not None:
                tol = max(0.03, max(4 * var[c]["mae_sd"] / var[c]["mae_mean"] for c in COLUMNS))
            else:
                tol = 0.10
            if level == 2:
                tol = max(tol, 0.15)
        mm = comp["metrics"]["model"]
        dev = {c: abs(mm[f"{c}_MAE"] - rm[f"{c}_MAE"]) / rm[f"{c}_MAE"] for c in COLUMNS}
        row |= {"model_MAE": {c: mm[f"{c}_MAE"] for c in COLUMNS},
                "reference_MAE": {c: rm[f"{c}_MAE"] for c in COLUMNS},
                "relative_deviation": dev, "tolerance": tol,
                "metrics_within_tolerance": all(v <= tol for v in dev.values()),
                "identical": all(v == 0.0 for v in dev.values())}
        if comp["vs"]:
            row["mae_ratio_vs_nf"] = {c: comp["vs"][f"NeuralFoil {PRIMARY_NF}"]["mae_ratio"][c] for c in COLUMNS}
            row["claim_holds"] = _claims_hold(comp)
            row["neuralfoil_version_matches"] = neuralfoil_version() == ref["neuralfoil"]["version"]
        else:
            row["claim_holds"] = None
        report["seeds"][seed] = row
    rows = report["seeds"].values()
    report["passed"] = all(r["metrics_within_tolerance"] and r.get("claim_holds") is not False
                           and r.get("files_ok", True) for r in rows)
    return report


def summarize_mains(results_dir: str | Path = Layout().results) -> dict:
    """Pool the per-point errors of all main replications: aggregate MAE ratios with bootstrap CIs."""
    files = sorted(Path(results_dir).glob("main_s*_pred.npz"))
    if not files:
        raise FileNotFoundError("no main_s*_pred.npz files")
    ea, eb, per_seed = [], [], {}
    for f in files:
        z = np.load(f)
        a, b = np.abs(z["model"] - z["y"]), np.abs(z[f"nf_{PRIMARY_NF}"] - z["y"])
        ea.append(a)
        eb.append(b)
        per_seed[f.stem.split("_")[1]] = (a.mean(0) / b.mean(0)).tolist()
    A, B = np.concatenate(ea), np.concatenate(eb)
    r, lo, hi = bootstrap_ratio(A, B, n_boot=2000)
    rr, rlo, rhi = bootstrap_ratio(A, B, n_boot=2000, stat="rms")
    return {"n_points": len(A), "n_replications": len(files),
            "pooled_mae_ratio": {c: [float(r[j]), float(lo[j]), float(hi[j])] for j, c in enumerate(COLUMNS)},
            "pooled_rmse_ratio": {c: [float(rr[j]), float(rlo[j]), float(rhi[j])] for j, c in enumerate(COLUMNS)},
            "per_seed_mae_ratio": per_seed,
            "pooled_mae": {"model": dict(zip(COLUMNS, map(float, A.mean(0)))),
                           "neuralfoil": dict(zip(COLUMNS, map(float, B.mean(0))))},
            "pooled_medae": {"model": dict(zip(COLUMNS, map(float, np.median(A, 0)))),
                             "neuralfoil": dict(zip(COLUMNS, map(float, np.median(B, 0))))},
            "ratio_sd_across_seeds": {c: float(np.std([v[j] for v in per_seed.values()], ddof=1))
                                      if len(per_seed) > 1 else math.nan for j, c in enumerate(COLUMNS)}}


# ------------------------------------------------------------------------------------------------ support study
def _auc(score: np.ndarray, positive: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney form)."""
    from scipy.stats import rankdata

    n1, n0 = int(positive.sum()), int((~positive).sum())
    if n1 == 0 or n0 == 0:
        return math.nan                                   # undefined when one class is empty
    r = rankdata(score)
    return float((r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def run_support_study(seeds: Sequence[int], layout: Layout = Layout(), n_airfoils: int = 2000,
                      n_conditions: int = 25) -> dict:
    """How well does the support flag (:mod:`neuralnaca.support`) identify where XFOIL fails, and what is lost?

    Per replication: the support model is built from the training set only and evaluated on the held-out test
    set - AUC for predicting XFOIL non-convergence, the share of converged / non-converged test points flagged, and
    the MAE of both models on the supported converged points. The share of the domain volume that is unsupported is
    estimated by Monte Carlo (``n_airfoils`` random sections x ``n_conditions`` random (Re, alpha) each).
    """
    from .domain import sample_points
    from .support import SupportModel

    rows = {}
    for seed in seeds:
        ref = json.loads((layout.results / f"main_s{seed}.json").read_text(encoding="utf-8"))
        tr = XfoilDataset.load(layout.data / ref["datasets"]["train"]["file"])
        te = XfoilDataset.load(layout.data / ref["datasets"]["test"]["file"])
        net, _ = load_model(layout.models / ref["model"]["file"])
        sup = SupportModel.from_dataset(tr)
        pts = te.points
        pred_all = predict_tensors(net, to_tensors(pts, airfoil_constants(pts[:, :3]))).mean(0)
        p_conv = sup.convergence_probability(pts)
        flag = ~sup.supported(pts, np.exp(pred_all[:, 1]))
        nonconv = ~te.converged
        ok = te.converged
        z = np.load(layout.results / f"main_s{seed}_pred.npz")
        keep = ~flag[ok]
        y = z["y"]
        ea, eb = np.abs(z["model"] - y), np.abs(z[f"nf_{PRIMARY_NF}"] - y)
        foils = sample_points(n_airfoils, "random", 99_000 + seed)
        cond = sample_points(n_airfoils * n_conditions, "random", 98_000 + seed)
        vol = np.concatenate([np.repeat(foils[:, :3], n_conditions, 0), cond[:, 3:]], 1)
        const = np.repeat(airfoil_constants(foils[:, :3]), n_conditions, 0)
        vp = predict_tensors(net, to_tensors(vol, const)).mean(0)
        # ensemble spread as an error indicator on the converged test points
        from scipy.stats import spearmanr

        conv_pts = pts[ok]
        members = predict_tensors(net, to_tensors(conv_pts, airfoil_constants(conv_pts[:, :3])))
        sd = members.std(0)
        spread = {}
        for j, c in enumerate(COLUMNS):
            top = np.argsort(-ea[:, j])[:20]
            spread[c] = {"spearman_std_vs_abs_error": float(spearmanr(sd[:, j], ea[:, j])[0]),
                         "worst20_in_top5pct_std": int((sd[top, j] > np.quantile(sd[:, j], 0.95)).sum()),
                         "median_abs_error_over_std": float(np.median(ea[:, j] / sd[:, j]))}
        rows[str(seed)] = {
            "ensemble_spread": spread,
            "auc_nonconvergence": _auc(1 - p_conv, nonconv),
            "flagged_converged_pct": float(100 * flag[ok].mean()),
            "flagged_nonconverged_pct": float(100 * flag[nonconv].mean()) if nonconv.any() else math.nan,
            "n_nonconverged": int(nonconv.sum()),
            "unsupported_volume_pct": float(100 * (~sup.supported(vol, np.exp(vp[:, 1]))).mean()),
            "max_CD_in_volume": float(np.exp(vp[:, 1]).max()),
            "mae_supported": {"model": dict(zip(COLUMNS, map(float, ea[keep].mean(0)))),
                              "neuralfoil": dict(zip(COLUMNS, map(float, eb[keep].mean(0))))},
            "mae_ratio_supported": dict(zip(COLUMNS, map(float, ea[keep].mean(0) / eb[keep].mean(0)))),
            "cd_max": sup.cd_max,
        }
    res = {"kind": "support", "seeds": list(seeds), "runs": rows, "environment": environment_info()}
    _write(res, layout.results / "support.json")
    return res


# ------------------------------------------------------------------------------------------------ reference polars
POLAR_SECTIONS = ("0012", "2412", "4412", "6409")
POLAR_RE = (1e6, 2e5)


def run_polar_reference(layout: Layout = Layout(), workers: int | None = None) -> Path:
    """XFOIL and NeuralFoil polars for the documentation figures (sections not in any training set).

    Stored so that the figures can be regenerated without XFOIL; NeuralNACA is evaluated live when plotting.
    """
    from .geometry import parse_naca
    from .xfoil import run_xfoil

    alphas = np.arange(-5.0, 15.01, 0.5)
    pts = np.array([[*parse_naca(c), re, a] for c in POLAR_SECTIONS for re in POLAR_RE for a in alphas])
    xf = run_xfoil(pts, detect_xfoil(), workers, "polars")
    nf = neuralfoil_predict(pts, PRIMARY_NF) if neuralfoil_version() else np.full((len(pts), 3), np.nan)
    path = layout.results / "polars_reference.npz"
    with open(path, "wb") as fh:
        np.savez(fh, points=pts, xfoil=xf, neuralfoil=nf, sections=np.array(POLAR_SECTIONS), re=np.array(POLAR_RE),
                 alphas=alphas)
    return path


# ------------------------------------------------------------------------------------------------ speed
def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def run_speed(layout: Layout = Layout(), repeats: int = 3) -> dict:
    """Wall-clock cost per condition of NeuralNACA, NeuralFoil-xxxlarge and XFOIL on this machine.

    Cases: one condition on a new airfoil each call; a 41-point polar of one airfoil; 1,000 conditions on 1,000
    different airfoils. Best of ``repeats``. Absolute numbers depend on the hardware; the ratios are the point.
    """
    import os

    import torch

    from .api import NeuralNACA
    from .domain import sample_points
    from .xfoil import run_xfoil

    model = NeuralNACA.pretrained()
    pts = sample_points(1000, "random", 4242)
    alphas = np.arange(-5.0, 15.01, 0.5)

    def best(fn):
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return min(times)

    def ours_single():
        for p in pts[:50]:
            NeuralNACA(model.net, model.meta, support=model.support).predict(
                m=p[0], p=p[1], t=p[2], Re=p[3], alpha=p[4], on_domain_error="ignore")

    def ours_polar():
        NeuralNACA(model.net, model.meta, support=model.support).predict(
            m=0.02, p=0.4, t=0.12, Re=1e6, alpha=alphas, on_domain_error="ignore")

    def ours_batch():
        NeuralNACA(model.net, model.meta, support=model.support).predict(
            m=pts[:, 0], p=pts[:, 1], t=pts[:, 2], Re=pts[:, 3], alpha=pts[:, 4], on_domain_error="ignore")

    res: dict = {"kind": "speed", "cpu": _cpu_name(), "cpus": os.cpu_count(), "torch_threads": torch.get_num_threads(),
                 "seconds_per_condition": {}}
    sp = res["seconds_per_condition"]
    sp["NeuralNACA"] = {"single (new airfoil each call)": best(ours_single) / 50,
                        "polar (41 angles, one airfoil)": best(ours_polar) / len(alphas),
                        "batch (1000 airfoils)": best(ours_batch) / len(pts)}
    if neuralfoil_version():
        import neuralfoil as nf

        from .geometry import naca4_coordinates

        def nf_single():
            for p in pts[:50]:
                nf.get_aero_from_coordinates(naca4_coordinates(*p[:3]), alpha=p[4], Re=p[3], model_size=PRIMARY_NF)

        def nf_polar():
            nf.get_aero_from_coordinates(naca4_coordinates(0.02, 0.4, 0.12), alpha=alphas, Re=1e6,
                                         model_size=PRIMARY_NF)

        def nf_batch():
            neuralfoil_predict(pts, PRIMARY_NF)

        sp[f"NeuralFoil {PRIMARY_NF}"] = {"single (new airfoil each call)": best(nf_single) / 50,
                                          "polar (41 angles, one airfoil)": best(nf_polar) / len(alphas),
                                          "batch (1000 airfoils)": best(nf_batch) / len(pts)}
    try:
        cfg = detect_xfoil()
        sub = pts[:40]
        t0 = time.perf_counter()
        run_xfoil(sub, cfg, workers=1, chunk=len(sub))
        t1 = time.perf_counter() - t0
        sp[f"XFOIL ({cfg.backend}, 1 process)"] = {"single (new airfoil each call)": t1 / len(sub)}
    except Exception as e:  # pragma: no cover - machine dependent
        res["xfoil_error"] = repr(e)
    res["environment"] = environment_info()
    _write(res, layout.results / "speed.json")
    return res
