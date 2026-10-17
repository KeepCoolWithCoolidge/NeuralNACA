"""Documentation figures, regenerated from the shipped results (``neuralnaca figures``)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .evaluation import COLUMNS

__all__ = ["make_figures"]

# Reference data-viz palette (validated categorical order; text in neutral ink, never in series colours)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
OURS, NF, REF = "#2a78d6", "#eb6834", "#0b0b0b"
SERIES3 = ("#2a78d6", "#eb6834", "#1baf7a")
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
YLAB = {"CL": "$C_L$", "lnCD": r"$\ln C_D$", "CM": "$C_M$"}


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
    })
    return plt


def fig_polars(root: Path, out: Path, re: float) -> Path:
    from .api import NeuralNACA

    plt = _style()
    z = np.load(Path(root) / "benchmarks" / "results" / "polars_reference.npz")
    pts, xf, nf, sections, alphas = z["points"], z["xfoil"], z["neuralfoil"], z["sections"], z["alphas"]
    model = NeuralNACA.pretrained()
    fig, axs = plt.subplots(len(sections), 3, figsize=(11, 2.6 * len(sections)), sharex=True)
    for i, code in enumerate(sections):
        sel = _is(pts, code) & np.isclose(pts[:, 3], re)
        ours = model.predict(str(code), alpha=alphas, Re=re, on_domain_error="ignore")
        for j, key in enumerate(("CL", "CD", "CM")):
            ax = axs[i, j]
            ax.plot(alphas, nf[sel, j] if j != 1 else np.exp(nf[sel, 1]), color=NF, lw=1.6, ls=(0, (4, 2)),
                    label="NeuralFoil xxxlarge")
            ax.plot(alphas, ours[key], color=OURS, lw=2.0, label="NeuralNACA")
            ax.plot(alphas, xf[sel, j], "o", ms=3.6, color=REF, mfc=SURFACE, mew=1.1, label="XFOIL", zorder=3)
            if key == "CD":
                ax.set_yscale("log")
            ax.set_ylabel(f"NACA {code}\n{YLAB.get(key, '$C_D$')}" if j == 0 else YLAB.get(key, "$C_D$"))
            if i == len(sections) - 1:
                ax.set_xlabel(r"$\alpha$ [deg]")
    axs[0, 0].legend(loc="upper left", fontsize=9)
    fig.suptitle(f"Polars at Re = {re:.0e}: sections absent from every training set "
                 f"(open circles: XFOIL; missing circles: XFOIL did not converge)", fontsize=10.5, color=INK)
    fig.tight_layout()
    path = out / f"polars_re{re:.0e}.png".replace("+", "")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _is(pts, code):
    from .geometry import parse_naca

    return np.all(np.isclose(pts[:, :3], parse_naca(str(code))), 1)


def fig_curve(root: Path, out: Path) -> Path | None:
    plt = _style()
    rows: dict[int, list] = {}
    for p in sorted((Path(root) / "benchmarks" / "results").glob("curve_s*.json")):
        for k, r in json.loads(p.read_text(encoding="utf-8"))["runs"].items():
            rows.setdefault(int(k), []).append(r)
    if not rows:
        return None
    ns = np.array(sorted(rows))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axhline(1.0, color=INK2, lw=1.2, ls=(0, (2, 2)))
    ax.text(ns[0], 1.02, "NeuralFoil xxxlarge (7.9 M XFOIL runs)", color=INK2, fontsize=9, va="bottom")
    ends = {}
    for c, col in zip(COLUMNS, SERIES3):
        q = [np.array([r["mae_ratio_vs_nf"][c][0] for r in rows[n]]) for n in ns]
        mean = np.array([v.mean() for v in q])
        ax.fill_between(ns, [v.min() for v in q], [v.max() for v in q], color=col, alpha=0.15, lw=0)
        ax.plot(ns, mean, "-o", color=col, ms=5, mec=SURFACE, mew=1.5, label=YLAB[c])
        ends[c] = (mean[-1], col)
    # direct labels at the line ends, spread apart so that they never overlap
    order = sorted(ends, key=lambda c: ends[c][0])
    ys = [ends[c][0] for c in order]
    gap = 0.07 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    for c, yl in zip(order, ys):
        ax.annotate(YLAB[c], (ns[-1], ends[c][0]), xytext=(ns[-1] * 1.12, yl), textcoords="data", va="center",
                    color=INK, fontsize=10, arrowprops={"arrowstyle": "-", "color": ends[c][1], "lw": 1})
    nmax = {n: len(rows[n]) for n in ns}
    if min(nmax.values()) != max(nmax.values()):
        ax.text(0.99, 0.97, "replications per point: " + ", ".join(f"{n:,}: {k}" for n, k in nmax.items()),
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([f"{n:,}" for n in ns])
    ax.set_xlabel("XFOIL runs used for training")
    ax.set_ylabel("MAE ratio  NeuralNACA / NeuralFoil")
    ax.set_title("Data efficiency (lines: mean over replications; bands: range)", fontsize=10.5)
    ax.set_xlim(ns[0] / 1.3, ns[-1] * 1.9)
    fig.tight_layout()
    path = out / "data_efficiency.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_ablation(root: Path, out: Path) -> Path | None:
    from .report import ABLATION_LABEL

    plt = _style()
    runs: dict[str, list] = {}
    for p in sorted((Path(root) / "benchmarks" / "results").glob("ablation_s*.json")):
        for k, r in json.loads(p.read_text(encoding="utf-8"))["runs"].items():
            runs.setdefault(k, []).append(r)
    order = [k for k in ("plain", "no_panel", "no_symmetry", "no_drag_prior", "full+residuals", "full") if k in runs]
    if not order:
        return None
    phys = ""
    if "physics_only" in runs:                    # far off the scale of the others; quoted in the title instead
        v = [np.mean([r["mae_ratio_vs_nf"][c][0] for r in runs["physics_only"]]) for c in COLUMNS]
        phys = f"; physics alone without a network: {min(v):.0f}-{max(v):.0f}x NeuralFoil's error (not shown)"
    fig, axs = plt.subplots(1, 3, figsize=(11, 0.55 * len(order) + 1.4), sharey=True)
    for ax, c in zip(axs, COLUMNS):
        vals = [np.mean([r["mae_ratio_vs_nf"][c][0] for r in runs[k]]) for k in order]
        y = np.arange(len(order))
        ax.barh(y, vals, height=0.6, color=[OURS if k == "full" else "#9ec5f4" for k in order], edgecolor=SURFACE,
                linewidth=2)
        ax.axvline(1.0, color=INK2, lw=1.2, ls=(0, (2, 2)))
        for yi, v in zip(y, vals):
            ax.text(v + 0.02, yi, f"{v:.2f}", va="center", fontsize=9, color=INK)
        ax.set_title(f"{YLAB[c]}: MAE / NeuralFoil MAE", fontsize=10.5)
        ax.set_xlim(0, max(1.15, max(vals) * 1.18))
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(np.arange(len(order)))
    nmax = max(len(runs[k]) for k in order)
    axs[0].set_yticklabels([ABLATION_LABEL[k].replace("**", "")
                            + (f" ({len(runs[k])} of {nmax} replications)" if len(runs[k]) < nmax else "")
                            for k in order])
    fig.suptitle(f"Ablation (mean over replications; dashed line = NeuralFoil parity{phys})", fontsize=10.5)
    fig.tight_layout()
    path = out / "ablation.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_error_cdf(root: Path, out: Path) -> Path | None:
    plt = _style()
    files = sorted((Path(root) / "benchmarks" / "results").glob("main_s*_pred.npz"))
    if not files:
        return None
    ea, eb = [], []
    for f in files:
        z = np.load(f)
        ea.append(np.abs(z["model"] - z["y"]))
        eb.append(np.abs(z["nf_xxxlarge"] - z["y"]))
    A, B = np.concatenate(ea), np.concatenate(eb)
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
    for j, (ax, c) in enumerate(zip(axs, COLUMNS)):
        for e, col, lab in ((B[:, j], NF, "NeuralFoil xxxlarge"), (A[:, j], OURS, "NeuralNACA")):
            s = np.sort(np.maximum(e, 1e-7))
            ax.plot(s, np.arange(1, len(s) + 1) / len(s), color=col, label=lab)
        ax.set_xscale("log")
        ax.set_xlabel(f"|error| of {YLAB[c]}")
        ax.set_ylim(0, 1.0)
        if j == 0:
            ax.set_ylabel("share of test points")
            ax.legend(loc="upper left", fontsize=9)
    fig.suptitle(f"Error distributions, all replications pooled ({len(A):,} XFOIL-converged test points)",
                 fontsize=10.5)
    fig.tight_layout()
    path = out / "error_cdf.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_support(root: Path, out: Path) -> Path:
    from matplotlib.colors import BoundaryNorm, ListedColormap

    from .api import NeuralNACA

    plt = _style()
    model = NeuralNACA.pretrained()
    a = np.linspace(-5, 15, 81)
    lre = np.linspace(5, 7, 61)
    A, L = np.meshgrid(a, lre)
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True, layout="constrained")
    cmap = ListedColormap(SEQ[:5])
    norm = BoundaryNorm([0, 0.2, 0.4, 0.5, 0.8, 1.0], cmap.N)
    im = None
    for ax, t in zip(axs, (0.06, 0.12, 0.18)):
        o = model.predict(m=0.02, p=0.4, t=t, alpha=A, Re=10 ** L, on_domain_error="ignore")
        im = ax.pcolormesh(A, L, o["xfoil_convergence"], cmap=cmap, norm=norm, shading="auto")
        ax.contour(A, L, o["supported"].astype(float), levels=[0.5], colors=INK, linewidths=1.6)
        ax.set_title(f"thickness t = {t:.2f}", fontsize=10)
        ax.set_xlabel(r"$\alpha$ [deg]")
        ax.grid(False)
    axs[0].set_ylabel(r"$\log_{10}$ Re")
    cb = fig.colorbar(im, ax=axs, shrink=0.9, pad=0.01)
    cb.set_label("estimated P(XFOIL converges)")
    fig.suptitle("Training support for m = 0.02, p = 0.4: predictions right of / below the black line are flagged "
                 "unsupported", fontsize=10.5)
    path = out / "support_map.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def make_figures(root: Path, out: Path) -> list[Path]:
    """Regenerate every documentation figure from the results under ``root``; returns the written paths."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    if (Path(root) / "benchmarks" / "results" / "polars_reference.npz").exists():
        paths += [fig_polars(root, out, 1e6), fig_polars(root, out, 2e5)]
    for f in (fig_curve, fig_ablation, fig_error_cdf):
        p = f(root, out)
        if p is not None:
            paths.append(p)
    paths.append(fig_support(root, out))
    return paths
