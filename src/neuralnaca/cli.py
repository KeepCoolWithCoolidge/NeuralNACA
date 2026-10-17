"""Command-line interface: ``neuralnaca <command> ...`` (also ``python -m neuralnaca``)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from ._version import __version__

log = logging.getLogger("neuralnaca")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(message)s", stream=sys.stderr)


def _threads(n: int | None) -> None:
    if n:
        import torch

        torch.set_num_threads(n)


def _seeds(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


# ------------------------------------------------------------------------------------------------ commands
def cmd_predict(a) -> int:
    from .api import NeuralNACA

    model = NeuralNACA.load(a.weights) if a.weights else NeuralNACA.pretrained()
    alphas = np.arange(a.alpha[0], a.alpha[1] + 1e-9, a.step) if len(a.alpha) == 2 else np.array(a.alpha)
    out = model.predict(a.airfoil, alpha=alphas, Re=a.re, on_domain_error="ignore")
    ref = None
    if a.compare_xfoil:
        from .geometry import parse_naca
        from .xfoil import detect_xfoil, run_xfoil

        m, p, t = parse_naca(a.airfoil)
        ref = run_xfoil(np.array([[m, p, t, a.re, al] for al in alphas]), detect_xfoil(), a.workers, "xfoil")
    if a.json:
        doc = {"airfoil": a.airfoil, "Re": a.re, "alpha": alphas.tolist(),
               **{k: v.tolist() for k, v in out.items()}}
        if ref is not None:
            doc["xfoil"] = {"CL": ref[:, 0].tolist(), "CD": ref[:, 1].tolist(), "CM": ref[:, 2].tolist()}
        print(json.dumps(doc, indent=1))
        return 0
    code = a.airfoil.upper().replace("NACA", "").strip()
    print(f"NACA {code}  Re = {a.re:.4g}  (n_crit = 9, Mach 0, free transition)")
    hdr = f"{'alpha':>7} {'CL':>8} {'CD':>9} {'CM':>8}   {'+-CL':>6} {'+-CD%':>6}"
    if ref is not None:
        hdr += f"   | XFOIL {'CL':>7} {'CD':>9} {'CM':>8}"
    print(hdr)
    def z(v, nd):  # round for display, without printing "-0.0000"
        return round(float(v), nd) + 0.0

    for i, al in enumerate(alphas):
        s = (f"{al:7.2f} {z(out['CL'][i], 4):8.4f} {out['CD'][i]:9.5f} {z(out['CM'][i], 4):8.4f}   "
             f"{out['CL_std'][i]:6.4f} {100 * out['lnCD_std'][i]:6.2f}")
        if ref is not None:
            s += f"   |       {ref[i, 0]:7.4f} {ref[i, 1]:9.5f} {ref[i, 2]:8.4f}"
        if not out["in_domain"][i]:
            s += "   (outside validated domain)"
        print(s)
    return 0


def cmd_info(a) -> int:
    from .api import NeuralNACA
    from .benchmark import environment_info
    from .domain import DEFAULT_DOMAIN

    model = NeuralNACA.pretrained()
    info = {"version": __version__, "model": repr(model), "domain": DEFAULT_DOMAIN.describe(),
            "training": {k: model.meta.get(k) for k in ("train_data", "n_train_runs", "n_train_converged")},
            "environment": environment_info()}
    try:
        from .xfoil import detect_xfoil, xfoil_version

        cfg = detect_xfoil()
        info["xfoil"] = xfoil_version(cfg)
    except Exception as e:  # pragma: no cover - depends on the machine
        info["xfoil"] = f"not available ({type(e).__name__})"
    print(json.dumps(info, indent=1))
    return 0


def cmd_xfoil_check(a) -> int:
    """Solve a few reference points and compare with values from the reference XFOIL 6.99 build."""
    from .xfoil import detect_xfoil, run_xfoil, xfoil_version

    cfg = detect_xfoil(a.backend, a.executable)
    print(f"backend: {cfg.backend}  ({xfoil_version(cfg)})")
    # NACA 2412 (0, 4 deg) and 0012 (5 deg) at Re = 1e6; cold start, 279 panels, n_crit 9. Reference values from the
    # XFOIL 6.99 builds used for the benchmark (python module and executable agree to the 4 printed decimals).
    pts = np.array([[0.02, 0.4, 0.12, 1e6, 0.0], [0.02, 0.4, 0.12, 1e6, 4.0], [0.0, 0.0, 0.12, 1e6, 5.0]])
    ref = np.array([[0.2412, 0.00563, -0.0517], [0.7168, 0.00702, -0.0566], [0.5563, 0.00850, 0.0021]])
    got = run_xfoil(pts, cfg, 1, "check")
    ok = bool(np.isfinite(got).all() and np.allclose(got, ref, rtol=0, atol=np.array([0.005, 0.0002, 0.003])))
    for g, r in zip(got, ref):
        print(f"  got CL {g[0]:.4f} CD {g[1]:.5f} CM {g[2]:.4f}   reference CL {r[0]:.4f} CD {r[1]:.5f} CM {r[2]:.4f}")
    print("XFOIL check:", "PASS" if ok else "FAIL - results differ from the reference build")
    return 0 if ok else 1


def cmd_generate(a) -> int:
    from .data import generate_dataset
    from .xfoil import detect_xfoil

    cfg = detect_xfoil(a.backend, a.executable, panels=a.panels, protocol=a.protocol, geometry=a.geometry,
                       max_iter=a.max_iter)
    ds = generate_dataset(a.n, a.kind, a.seed, cfg, a.workers, path=a.out, label="generate")
    print(f"wrote {a.out}: {ds.converged.sum()}/{len(ds)} converged")
    return 0


def cmd_train(a) -> int:
    from .benchmark import environment_info
    from .data import XfoilDataset, pack_dataset, sha256
    from .model import check_reflection, save_model
    from .training import TrainConfig, train_ensemble

    ds = XfoilDataset.load(a.data)
    cfg = TrainConfig(ensemble=a.ensemble, steps=a.steps, seed=a.seed, variant=a.variant)
    net = train_ensemble(pack_dataset(ds), cfg)
    meta = {"train_data": Path(a.data).name, "train_data_sha256": sha256(a.data), "n_train_runs": len(ds),
            "n_train_converged": int(ds.converged.sum()), "train_config": cfg.to_dict(),
            "environment": environment_info()}
    save_model(net, a.out, meta)
    from .support import SupportModel

    sup_path = Path(a.out).with_suffix(".support.npz")
    SupportModel.from_dataset(ds).save(sup_path)
    print(f"wrote {a.out} and {sup_path}; reflection check: {check_reflection(net)}")
    return 0


def cmd_evaluate(a) -> int:
    from .benchmark import _nf_predictions, _predict
    from .data import XfoilDataset
    from .model import load_model
    from .report import format_comparison

    net, _ = load_model(a.weights)
    te = XfoilDataset.load(a.data)
    sizes = [s for s in a.neuralfoil.split(",") if s]
    from .evaluation import compare

    nf = _nf_predictions(te.points[te.converged], sizes) if sizes else {}
    comp = compare(_predict(net, te), nf, te.targets(), te.points[te.converged])
    print(format_comparison(comp))
    if a.out:
        Path(a.out).write_text(json.dumps(comp, indent=1), encoding="utf-8")
    return 0


def cmd_bench(a) -> int:
    from . import benchmark as b

    layout = b.Layout(Path(a.root))
    spec = b.BenchmarkSpec()
    seeds = _seeds(a.seeds)
    if a.experiment == "support":
        b.run_support_study(seeds, layout)
        return 0
    if a.experiment == "speed":
        b.run_speed(layout)
        return 0
    for s in seeds:
        if a.experiment == "main":
            b.run_main(s, layout, spec, a.workers, retrain=a.retrain)
        elif a.experiment == "variance":
            b.run_variance(s, layout=layout, spec=spec, retrain=a.retrain)
        elif a.experiment == "ablation":
            b.run_ablation(s, layout=layout, spec=spec, residuals=a.residuals, retrain=a.retrain,
                           variants=a.variants.split(",") if a.variants else b.ABLATION_VARIANTS)
        elif a.experiment == "curve":
            b.run_curve(s, sizes=[int(v) for v in a.sizes.split(",")], layout=layout, spec=spec, workers=a.workers,
                        retrain=a.retrain)
        elif a.experiment == "protocol":
            b.run_protocol_study(s, layout, spec, a.workers)
        log.info("finished %s seed %d", a.experiment, s)
    return 0


def cmd_reproduce(a) -> int:
    from . import benchmark as b
    from .report import format_reproduction

    out = b.Layout(Path(a.out)) if a.out else None
    rep = b.reproduce(a.level, _seeds(a.seeds) if a.seeds else None, b.Layout(Path(a.root)), out, a.workers)
    print(format_reproduction(rep))
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    return 0 if rep["passed"] else 1


def cmd_report(a) -> int:
    from .report import write_report

    path = write_report(Path(a.root), Path(a.out))
    print(f"wrote {path}")
    return 0


def cmd_figures(a) -> int:
    from .plotting import make_figures

    for p in make_figures(Path(a.root), Path(a.out)):
        print(f"wrote {p}")
    return 0


# ------------------------------------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="neuralnaca", description=__doc__)
    ap.add_argument("--version", action="version", version=f"neuralnaca {__version__}")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--threads", type=int, default=None, help="PyTorch CPU threads")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict", help="predict a polar with the shipped (or given) model")
    p.add_argument("airfoil", help="NACA 4-digit designation, e.g. 2412")
    p.add_argument("--re", type=float, required=True, help="Reynolds number")
    p.add_argument("--alpha", type=float, nargs="+", default=[-4.0, 14.0],
                   help="two values = range (with --step), otherwise the list of angles [deg]")
    p.add_argument("--step", type=float, default=1.0)
    p.add_argument("--weights", help="weights file (default: shipped model)")
    p.add_argument("--compare-xfoil", action="store_true", help="also run XFOIL (needs XFOIL)")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("info", help="version, shipped model, domain, environment, XFOIL availability")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("xfoil-check", help="check that your XFOIL reproduces the reference build")
    p.add_argument("--backend", default="auto", choices=["auto", "module", "binary"])
    p.add_argument("--executable", default=None)
    p.set_defaults(func=cmd_xfoil_check)

    p = sub.add_parser("generate", help="sample a design and run XFOIL on it")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--kind", choices=["sobol", "random"], default="sobol")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--backend", default="auto", choices=["auto", "module", "binary"])
    p.add_argument("--executable", default=None)
    p.add_argument("--panels", type=int, default=279)
    p.add_argument("--protocol", default="cold", choices=["cold", "nf-sweep"])
    p.add_argument("--geometry", default="naca", choices=["naca", "kulfan", "naca-normalized"])
    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument("--workers", type=int, default=None)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("train", help="train a model on an XFOIL data set")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ensemble", type=int, default=5)
    p.add_argument("--steps", type=int, default=18000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--variant", default="full", choices=["full", "no_symmetry", "no_panel", "no_drag_prior", "plain"])
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="evaluate a model on an XFOIL test set (optionally vs NeuralFoil)")
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--neuralfoil", default="xxxlarge", help="comma-separated NeuralFoil sizes ('' to skip)")
    p.add_argument("--out", default=None, help="write the comparison as JSON")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("bench", help="run a benchmark experiment (writes into --root)")
    p.add_argument("experiment", choices=["main", "variance", "ablation", "curve", "protocol", "support", "speed"])
    p.add_argument("--seeds", default="0", help="e.g. 0,1,2 or 0-4")
    p.add_argument("--root", default=".")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--residuals", action="store_true", help="ablation: also train full+residuals")
    p.add_argument("--variants", default=None, help="ablation: comma-separated subset of variants")
    p.add_argument("--sizes", default="512,1024,2048,4096", help="curve: training-set sizes")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("reproduce", help="re-derive the shipped results (0 = verify, 1 = retrain, 2 = regenerate)")
    p.add_argument("--level", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--seeds", default=None, help="default: all shipped replications")
    p.add_argument("--root", default=".", help="repository root with data/ and benchmarks/")
    p.add_argument("--out", default=None, help="levels 1-2: output root (default: <root>/reproduction)")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--json", default=None, help="also write the report as JSON")
    p.set_defaults(func=cmd_reproduce)

    p = sub.add_parser("report", help="regenerate the Markdown results tables from the JSON results")
    p.add_argument("--root", default=".")
    p.add_argument("--out", default="docs/results/tables.md")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("figures", help="regenerate the documentation figures from the shipped results")
    p.add_argument("--root", default=".")
    p.add_argument("--out", default="docs/assets")
    p.set_defaults(func=cmd_figures)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    _setup_logging(a.verbose)
    _threads(a.threads)
    try:
        return int(a.func(a) or 0)
    except Exception as e:
        from .xfoil import XfoilNotFoundError

        if isinstance(e, XfoilNotFoundError):
            print(str(e), file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
