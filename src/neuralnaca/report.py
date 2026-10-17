"""Human-readable reports generated from the JSON results (no number in the documentation is typed by hand)."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import numpy as np

from .baselines import NEURALFOIL_SIZES, NF_TRAIN_RUNS
from .evaluation import ALPHA_EDGES, COLUMNS, RE_EDGES

__all__ = ["build_headline", "build_report", "format_comparison", "format_reproduction", "write_report"]

LABEL = {"CL": "C<sub>L</sub>", "lnCD": "ln C<sub>D</sub>", "CM": "C<sub>M</sub>"}
PRIMARY = "NeuralFoil xxxlarge"


def _ci(v, digits=2) -> str:
    return f"{v[0]:.{digits}f} [{v[1]:.{digits}f}, {v[2]:.{digits}f}]"


def _verdict(v) -> str:
    return "better" if v[2] < 1 else ("worse" if v[1] > 1 else "n.s.")


# ------------------------------------------------------------------------------------------------ console text
def format_comparison(comp: dict) -> str:
    """Plain-text table of a :func:`neuralnaca.evaluation.compare` result."""
    lines = [f"{comp['n_test']} XFOIL-converged test points", ""]
    hdr = (f"{'model':24s} {'CL MAE':>8} {'lnCD MAE':>9} {'CM MAE':>9} | "
           f"{'CL MedAE':>9} {'lnCD MedAE':>10} {'CM MedAE':>9}")
    lines += [hdr, "-" * len(hdr)]
    for name, m in comp["metrics"].items():
        lines.append(f"{name:24s} {m['CL_MAE']:8.4f} {m['lnCD_MAE']:9.4f} {m['CM_MAE']:9.5f} | "
                     f"{m['CL_MedAE']:9.5f} {m['lnCD_MedAE']:10.5f} {m['CM_MedAE']:9.6f}")
    for name, vs in comp["vs"].items():
        lines += ["", f"MAE ratio model / {name} (95% paired bootstrap CI; < 1 = model more accurate)"]
        for c in COLUMNS:
            lines.append(f"  {c:5s} {_ci(vs['mae_ratio'][c])}  {_verdict(vs['mae_ratio'][c])}   "
                         f"RMSE ratio {_ci(vs['rmse_ratio'][c])}   Wilcoxon p = {vs['wilcoxon_p'][c]:.1e}")
    return "\n".join(lines)


def format_reproduction(rep: dict) -> str:
    names = {0: "verify shipped weights", 1: "retrain from shipped data", 2: "regenerate data with your XFOIL"}
    lines = [f"Reproduction level {rep['level']} ({names[rep['level']]})", ""]
    for seed, r in rep["seeds"].items():
        dev = ", ".join(f"{c} {100 * r['relative_deviation'][c]:.3f}%" for c in COLUMNS)
        flag = "identical" if r["identical"] else ("within tolerance" if r["metrics_within_tolerance"] else
                                                   "OUTSIDE TOLERANCE")
        lines.append(f"seed {seed}: MAE deviation from reference: {dev} (tolerance {100 * r['tolerance']:.2f}%) "
                     f"-> {flag}")
        if "files_ok" in r:
            lines.append(f"         shipped files match their SHA-256: {r['files_ok']}")
        if r.get("mae_ratio_vs_nf"):
            q = "  ".join(f"{c} {_ci(r['mae_ratio_vs_nf'][c])}" for c in COLUMNS)
            lines.append(f"         MAE ratio vs NeuralFoil xxxlarge: {q} -> claim holds: {r['claim_holds']}")
            if not r.get("neuralfoil_version_matches", True):
                lines.append("         note: installed NeuralFoil version differs from the reference run")
        else:
            lines.append("         (neuralfoil not installed: comparison skipped)")
        if "data_agreement" in r:
            lines.append(f"         data agreement with shipped XFOIL results: {r['data_agreement']}")
    lines += ["", "PASSED" if rep["passed"] else "FAILED"]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ markdown report
def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _main_section(res_dir: Path) -> list[str]:
    from .benchmark import summarize_mains

    mains = {int(p.stem.split("_s")[1]): json.loads(p.read_text())
             for p in sorted(res_dir.glob("main_s*.json")) if p.stem.count("_") == 1}
    if not mains:
        return []
    s = summarize_mains(res_dir)
    n_train = mains[min(mains)]["datasets"]["train"]["n"]
    out = ["## Headline: all replications pooled", "",
           f"{s['n_replications']} independent replications (each with its own Sobol training set of {n_train:,} "
           f"XFOIL runs = {100 * n_train / NF_TRAIN_RUNS:.3f} % of NeuralFoil's {NF_TRAIN_RUNS:,}, and its own i.i.d. "
           f"test set of unseen airfoils); {s['n_points']:,} XFOIL-converged test points in total. Ratios are "
           "NeuralNACA / NeuralFoil-xxxlarge with 95 % paired-bootstrap intervals (< 1: NeuralNACA more accurate).", "",
           "| output | NeuralNACA MAE | NeuralFoil MAE | MAE ratio | RMSE ratio | NeuralNACA median "
           "| NeuralFoil median |",
           "|---|---|---|---|---|---|---|"]
    for c in COLUMNS:
        d = 5 if c == "CM" else 4
        out.append(f"| {LABEL[c]} | {s['pooled_mae']['model'][c]:.{d}f} | {s['pooled_mae']['neuralfoil'][c]:.{d}f} | "
                   f"{_ci(s['pooled_mae_ratio'][c])} | {_ci(s['pooled_rmse_ratio'][c])} | "
                   f"{s['pooled_medae']['model'][c]:.{d + 1}f} | {s['pooled_medae']['neuralfoil'][c]:.{d + 1}f} |")
    out += ["", "## Per replication", "",
            "| seed | test pts | train conv. | C<sub>L</sub> ratio | ln C<sub>D</sub> ratio | C<sub>M</sub> ratio | "
            "RMSE ratios (C<sub>L</sub> / ln C<sub>D</sub> / C<sub>M</sub>) | Wilcoxon p (max) | reflection test |",
            "|---|---|---|---|---|---|---|---|---|"]
    for seed, r in mains.items():
        vs = r["comparison"]["vs"].get(PRIMARY)
        if not vs:
            continue
        rm = " / ".join(f"{vs['rmse_ratio'][c][0]:.2f}" for c in COLUMNS)
        pmax = max(vs["wilcoxon_p"].values())
        rc = r["reflection_check"]
        refl = f"pass ({max(rc['max_CL'], rc['max_lnCD'], rc['max_CM']):.0e})" if rc["ok"] else "FAIL"
        out.append(f"| {seed} | {r['comparison']['n_test']:,} | {r['datasets']['train']['converged']:,} | "
                   + " | ".join(_ci(vs["mae_ratio"][c]) for c in COLUMNS) + f" | {rm} | {pmax:.0e} | {refl} |")
    sd = s["ratio_sd_across_seeds"]
    out += ["", "Standard deviation of the MAE ratio across replications: "
            + ", ".join(f"{LABEL[c]} {sd[c]:.3f}" for c in COLUMNS) + "."]
    # NeuralFoil sizes
    sizes = [z for z in NEURALFOIL_SIZES
             if all(f"NeuralFoil {z}" in r["comparison"]["metrics"] for r in mains.values())]
    if sizes:
        out += ["", "## Against every NeuralFoil model size", "",
                "Mean over replications of the MAE. The last columns give NeuralNACA's MAE ratio against that "
                "size.", "",
                "| model | C<sub>L</sub> MAE | ln C<sub>D</sub> MAE | C<sub>M</sub> MAE | ratio C<sub>L</sub> | "
                "ratio ln C<sub>D</sub> | ratio C<sub>M</sub> |", "|---|---|---|---|---|---|---|"]
        ours = {c: np.mean([r["comparison"]["metrics"]["model"][f"{c}_MAE"] for r in mains.values()]) for c in COLUMNS}
        out.append(f"| **NeuralNACA** (this work) | {ours['CL']:.4f} | {ours['lnCD']:.4f} | {ours['CM']:.5f} "
                   "| - | - | - |")
        for z in sizes:
            v = {c: np.mean([r["comparison"]["metrics"][f"NeuralFoil {z}"][f"{c}_MAE"] for r in mains.values()])
                 for c in COLUMNS}
            q = {c: np.mean([r["comparison"]["vs"][f"NeuralFoil {z}"]["mae_ratio"][c][0] for r in mains.values()])
                 for c in COLUMNS}
            out.append(f"| NeuralFoil {z} | {v['CL']:.4f} | {v['lnCD']:.4f} | {v['CM']:.5f} | {q['CL']:.2f} | "
                       f"{q['lnCD']:.2f} | {q['CM']:.2f} |")
    # regime breakdown (pooled over replications)
    pts, ea, eb = [], [], []
    for f in sorted(res_dir.glob("main_s*_pred.npz")):
        z = np.load(f)
        pts.append(z["points"])
        ea.append(np.abs(z["model"] - z["y"]))
        eb.append(np.abs(z["nf_xxxlarge"] - z["y"]))
    P, A, B = np.concatenate(pts), np.concatenate(ea), np.concatenate(eb)
    out += ["", "## Where the errors are (all replications pooled)", "",
            "MAE of NeuralNACA vs NeuralFoil-xxxlarge per band.", "",
            "| band | points | C<sub>L</sub> | ln C<sub>D</sub> | C<sub>M</sub> |", "|---|---|---|---|---|"]
    for name, col, edges in (("&alpha; [deg]", 4, ALPHA_EDGES), ("Re", 3, RE_EDGES)):
        for lo, hi in pairwise(edges):
            k = (P[:, col] >= lo) & (P[:, col] < hi)
            a, b = A[k].mean(0), B[k].mean(0)
            band = f"{name} {lo:g} to {min(hi, 15.0):g}" if col == 4 else f"Re {lo:.0e} to {min(hi, 1e7):.0e}"
            out.append(f"| {band} | {k.sum():,} | {a[0]:.4f} vs {b[0]:.4f} | {a[1]:.4f} vs {b[1]:.4f} | "
                       f"{a[2]:.5f} vs {b[2]:.5f} |")
    return out


def _variance_section(res_dir: Path) -> list[str]:
    v = _load(res_dir / "variance_s0.json")
    if not v:
        return []
    out = ["", "## Optimisation variance (same data, different training seeds)", "",
           f"Replication 0's training set, retrained with {len(v['runs'])} training seeds "
           f"({', '.join(map(str, v['train_seeds']))}).", "",
           "| output | MAE mean | MAE sd | MAE ratio vs NeuralFoil: mean | sd | worst |", "|---|---|---|---|---|---|"]
    for c in COLUMNS:
        s = v["summary"][c]
        d = 5 if c == "CM" else 4
        out.append(f"| {LABEL[c]} | {s['mae_mean']:.{d}f} | {s['mae_sd']:.{d + 1}f} | {s['ratio_mean']:.3f} | "
                   f"{s['ratio_sd']:.3f} | {s['ratio_max']:.3f} |")
    return out


ABLATION_LABEL = {
    "physics_only": "physics alone (panel method + drag prior, no network)",
    "plain": "plain MLP (no physics)", "no_panel": "without panel-method backbone",
    "no_symmetry": "without mirror symmetry", "no_drag_prior": "without flat-plate drag prior",
    "full": "**full model**", "full+residuals": "full + autograd PINN residuals",
}


def _ablation_section(res_dir: Path) -> list[str]:
    runs: dict[str, dict[int, dict]] = {}
    for p in sorted(res_dir.glob("ablation_s*.json")):
        a = json.loads(p.read_text())
        for k, r in a["runs"].items():
            runs.setdefault(k, {})[a["seed"]] = r
    if not runs:
        return []

    def change(k: str) -> str:
        """Relative MAE change against the full model, paired on the replications both were run on."""
        seeds = sorted(set(runs[k]) & set(runs.get("full", {})))
        if k == "full" or not seeds:
            return "-"
        parts = []
        for c in COLUMNS:
            q = np.mean([runs[k][s]["metrics"][f"{c}_MAE"] for s in seeds]) / \
                np.mean([runs["full"][s]["metrics"][f"{c}_MAE"] for s in seeds])
            parts.append(f"{q:.0f}x" if q > 3 else f"{round(100 * (q - 1)) + 0:+d}%")
        return " / ".join(parts)

    out = ["", "## Ablation: what each physics ingredient contributes", "",
           "Same training data, same recipe, same test points; MAE averaged over the replications listed. "
           "The change is relative to the full model on the same replications (positive = worse without the "
           "ingredient).", "",
           "| variant | replications | C<sub>L</sub> MAE | ln C<sub>D</sub> MAE | C<sub>M</sub> MAE | vs full | "
           "beats NeuralFoil (all 3, CI) |", "|---|---|---|---|---|---|---|"]
    for k in ["physics_only", "plain", "no_panel", "no_symmetry", "no_drag_prior", "full", "full+residuals"]:
        if k not in runs:
            continue
        rs = list(runs[k].values())
        m = {c: np.mean([r["metrics"][f"{c}_MAE"] for r in rs]) for c in COLUMNS}
        wins = sum(all(r["mae_ratio_vs_nf"][c][2] < 1 for c in COLUMNS) for r in rs)
        reps = ", ".join(str(s) for s in sorted(runs[k]))
        out.append(f"| {ABLATION_LABEL.get(k, k)} | {reps} | {m['CL']:.4f} | {m['lnCD']:.4f} | {m['CM']:.5f} | "
                   f"{change(k)} | {wins}/{len(rs)} |")
    if "full" in runs and "full+residuals" in runs:
        shared = sorted(set(runs["full"]) & set(runs["full+residuals"]))
        if shared:
            t_full = np.mean([runs["full"][s]["train_minutes"] for s in shared])
            t_res = np.mean([runs["full+residuals"][s]["train_minutes"] for s in shared])
            out += ["", f"Training time on the reference machine (replication {', '.join(map(str, shared))}): "
                    f"{t_full:.1f} min for the full model, {t_res:.1f} min with the autograd residuals "
                    f"({t_res / t_full:.1f}x)."]
    return out


def _curve_section(res_dir: Path) -> list[str]:
    rows: dict[int, list[dict]] = {}
    for p in sorted(res_dir.glob("curve_s*.json")):
        for k, r in json.loads(p.read_text())["runs"].items():
            rows.setdefault(int(k), []).append(r)
    if not rows:
        return []
    out = ["", "## Data efficiency", "",
           "Same recipe trained on the first *n* points of each replication's Sobol training sequence. "
           "Ratios are NeuralNACA / NeuralFoil-xxxlarge MAE (mean over the replications available for that *n*).", "",
           "| XFOIL runs | share of NeuralFoil's | replications | C<sub>L</sub> MAE | ln C<sub>D</sub> MAE | "
           "C<sub>M</sub> MAE | ratio C<sub>L</sub> | ratio ln C<sub>D</sub> | ratio C<sub>M</sub> |",
           "|---|---|---|---|---|---|---|---|---|"]
    for n in sorted(rows):
        rs = rows[n]
        m = {c: np.mean([r["metrics"][f"{c}_MAE"] for r in rs]) for c in COLUMNS}
        q = {c: np.mean([r["mae_ratio_vs_nf"][c][0] for r in rs]) for c in COLUMNS}
        out.append(f"| {n:,} | {100 * n / NF_TRAIN_RUNS:.3f} % | {len(rs)} | {m['CL']:.4f} | {m['lnCD']:.4f} | "
                   f"{m['CM']:.5f} | {q['CL']:.2f} | {q['lnCD']:.2f} | {q['CM']:.2f} |")
    return out


def _protocol_section(res_dir: Path) -> list[str]:
    p = _load(res_dir / "protocol_s0.json")
    if not p:
        return []
    out = ["", "## Sensitivity to the definition of the XFOIL ground truth", "",
           f"Replication {p['seed']}'s test points re-solved under {len(p['truths'])} definitions; scored on the "
           f"{p['n_common']:,} points that converged under all of them. *naca* = exact NACA coordinates; *kulfan* = "
           "the 8+8-weight Kulfan fit that NeuralFoil analyses internally (in NeuralFoil's normalised frame); "
           "*naca-normalized* = exact coordinates in that same frame (control). NeuralNACA was trained on "
           "*cold/naca* labels and predicts XFOIL for the exact section; NeuralFoil's training labels were "
           "*nf-sweep/kulfan*.", "",
           "| ground truth | NeuralNACA MAE (C<sub>L</sub> / ln C<sub>D</sub> / C<sub>M</sub>) | "
           "NeuralFoil MAE | ratio C<sub>L</sub> | ratio ln C<sub>D</sub> | ratio C<sub>M</sub> |",
           "|---|---|---|---|---|---|"]
    for name, t in p["truths"].items():
        a, b = t["model"], t["neuralfoil"]
        out.append(f"| {name} | {a['CL_MAE']:.4f} / {a['lnCD_MAE']:.4f} / {a['CM_MAE']:.5f} | "
                   f"{b['CL_MAE']:.4f} / {b['lnCD_MAE']:.4f} / {b['CM_MAE']:.5f} | "
                   + " | ".join(_ci(t["mae_ratio"][c]) for c in COLUMNS) + " |")
    out += ["", "XFOIL against itself under different definitions (same points):", "",
            "| pair | C<sub>L</sub> MAE | ln C<sub>D</sub> MAE | C<sub>M</sub> MAE | C<sub>L</sub> median | "
            "share with &#124;&Delta;C<sub>L</sub>&#124; > 0.01 |", "|---|---|---|---|---|---|"]
    for name, d in p["xfoil_vs_xfoil"].items():
        out.append(f"| {name} | {d['CL_MAE']:.4f} | {d['lnCD_MAE']:.4f} | {d['CM_MAE']:.5f} | {d['CL_MedAE']:.5f} | "
                   f"{100 * d['frac_CL_diff_gt_0.01']:.1f} % |")
    return out


def _support_section(res_dir: Path) -> list[str]:
    s = _load(res_dir / "support.json")
    if not s:
        return []
    runs = s["runs"]
    out = ["", "## Knowing when not to trust a prediction", "",
           "Built from each replication's *training* set only and evaluated on its held-out test set "
           "(see the limitations page).", "",
           "| seed | AUC for XFOIL non-convergence | non-converged test pts flagged | converged test pts flagged | "
           "unsupported share of domain | MAE ratio on supported pts (C<sub>L</sub> / ln C<sub>D</sub> / "
           "C<sub>M</sub>) |", "|---|---|---|---|---|---|"]
    for seed, r in runs.items():
        q = " / ".join(f"{r['mae_ratio_supported'][c]:.2f}" for c in COLUMNS)
        out.append(f"| {seed} | {r['auc_nonconvergence']:.3f} | {r['flagged_nonconverged_pct']:.0f} % of "
                   f"{r['n_nonconverged']} | {r['flagged_converged_pct']:.2f} % | "
                   f"{r['unsupported_volume_pct']:.1f} % | "
                   f"{q} |")
    if all("ensemble_spread" in r for r in runs.values()):
        out += ["", "Ensemble spread as an error indicator (converged test points):", "",
                "| seed | Spearman(std, &#124;error&#124;) C<sub>L</sub> / ln C<sub>D</sub> / C<sub>M</sub> | "
                "of the 20 worst C<sub>L</sub> errors, how many have C<sub>L</sub>_std in the top 5 % | "
                "median &#124;error&#124; / std (C<sub>L</sub>) |", "|---|---|---|---|"]
        for seed, r in runs.items():
            e = r["ensemble_spread"]
            rho = " / ".join(f"{e[c]['spearman_std_vs_abs_error']:.2f}" for c in COLUMNS)
            out.append(f"| {seed} | {rho} | {e['CL']['worst20_in_top5pct_std']} / 20 | "
                       f"{e['CL']['median_abs_error_over_std']:.2f} |")
    return out


def _speed_section(res_dir: Path) -> list[str]:
    s = _load(res_dir / "speed.json")
    if not s:
        return []
    cases = ["single (new airfoil each call)", "polar (41 angles, one airfoil)", "batch (1000 airfoils)"]
    out = ["", "## Cost per prediction", "",
           f"Wall-clock milliseconds per condition on {s['cpu']} ({s['cpus']} CPUs, {s['torch_threads']} PyTorch "
           "threads), best of three runs. XFOIL: python module, one worker process, cold start, including process "
           "start-up.", "", "| tool | " + " | ".join(cases) + " |", "|---|---|---|---|"]
    for tool, t in s["seconds_per_condition"].items():
        out.append(f"| {tool} | " + " | ".join(f"{1e3 * t[c]:.2f}" if c in t else "-" for c in cases) + " |")
    return out


def _environment_section(res_dir: Path) -> list[str]:
    r = _load(res_dir / "main_s0.json")
    if not r:
        return []
    e = r["environment"]
    keys = ["neuralnaca", "python", "platform", "numpy", "scipy", "torch", "torch_threads", "neuralfoil", "aerosandbox"]
    return ["", "## Reference environment", "", "| component | version |", "|---|---|",
            *[f"| {k} | {e.get(k)} |" for k in keys]]


def build_report(root: Path) -> str:
    res_dir = Path(root) / "benchmarks" / "results"
    parts = ["<!-- generated by `neuralnaca report`; do not edit by hand -->", ""]
    for sec in (_main_section, _variance_section, _ablation_section, _curve_section, _protocol_section,
                _support_section, _speed_section, _environment_section):
        parts += sec(res_dir)
    return "\n".join(parts).rstrip() + "\n"


def build_headline(root: Path) -> str:
    """Only the pooled headline table (included on the documentation home page)."""
    lines = _main_section(Path(root) / "benchmarks" / "results")
    if not lines:
        return ""
    end = lines.index("## Per replication") if "## Per replication" in lines else len(lines)
    body = [l for l in lines[2:end] if l.strip()]
    return "<!-- generated by `neuralnaca report`; do not edit by hand -->\n\n" + "\n\n".join(
        [body[0], "\n".join(body[1:])]) + "\n"


def write_report(root: Path, out: Path) -> Path:
    """Write the full tables to ``out`` and the headline table to ``headline.md`` next to it."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_report(root))
    (out.parent / "headline.md").write_text(build_headline(root))
    return out
