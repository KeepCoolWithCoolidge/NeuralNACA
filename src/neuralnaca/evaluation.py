"""Error metrics, paired-bootstrap comparisons and regime breakdowns.

All metrics follow NeuralFoil's paper: absolute errors of ``CL``, ``ln(CD)`` and ``CM`` (``ln CD`` errors are
roughly relative drag errors: 0.01 in ``ln CD`` ~ 1 % in ``CD``). Every comparison is *paired* - both models are
scored on exactly the same XFOIL-converged points - and uncertainties come from a paired bootstrap over test points.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np

__all__ = ["COLUMNS", "bootstrap_ratio", "compare", "error_metrics", "regime_breakdown", "wilcoxon_signed_rank"]

COLUMNS = ("CL", "lnCD", "CM")
ALPHA_EDGES = (-5.0, 0.0, 5.0, 10.0, 15.01)
RE_EDGES = (1e5, 3e5, 1e6, 3e6, 1e7 + 1)


def error_metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Summary statistics of the errors of ``pred`` against ``y`` (both ``(n, 3)``: ``CL, ln CD, CM``).

    Returns MAE, RMSE, median absolute error (MedAE) and 95th percentile (P95) per output, plus the mean relative
    drag error ``CD_rel_pct`` and absolute-CD MAE / RMSE.
    """
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    if pred.shape != y.shape or pred.ndim != 2 or pred.shape[1] != 3:
        raise ValueError("pred and y must both have shape (n, 3)")
    if not np.isfinite(pred).all():
        raise ValueError("predictions contain non-finite values")
    e = np.abs(pred - y)
    cd_p, cd_t = np.exp(pred[:, 1]), np.exp(y[:, 1])
    out: dict[str, float] = {}
    for j, c in enumerate(COLUMNS):
        out[f"{c}_MAE"] = float(e[:, j].mean())
        out[f"{c}_RMSE"] = float(math.sqrt(np.mean(e[:, j] ** 2)))
        out[f"{c}_MedAE"] = float(np.median(e[:, j]))
        out[f"{c}_P95"] = float(np.percentile(e[:, j], 95))
    out["CD_rel_pct"] = float(100 * np.mean(np.abs(cd_p - cd_t) / cd_t))
    out["CD_MAE"] = float(np.abs(cd_p - cd_t).mean())
    out["CD_RMSE"] = float(math.sqrt(np.mean((cd_p - cd_t) ** 2)))
    return out


def bootstrap_ratio(e_a: np.ndarray, e_b: np.ndarray, n_boot: int = 2000, seed: int = 0,
                    stat: str = "mean") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Paired bootstrap of ``stat(e_a) / stat(e_b)`` per column.

    Parameters
    ----------
    e_a, e_b
        ``(n, k)`` absolute errors of two models on the same points.
    n_boot
        Bootstrap resamples.
    seed
        Seed of the resampling (fixed -> reproducible intervals).
    stat
        ``"mean"`` (ratio of MAEs) or ``"rms"`` (ratio of RMSEs).

    Returns
    -------
    (ratio, lo95, hi95)
        Point estimate and percentile 95 % confidence interval per column.
    """
    e_a, e_b = np.asarray(e_a, float), np.asarray(e_b, float)
    if e_a.shape != e_b.shape:
        raise ValueError("paired errors must have the same shape")
    rng = np.random.default_rng(seed)
    n = len(e_a)
    idx = rng.integers(0, n, (n_boot, n))
    if stat == "mean":
        def f(e, i=slice(None)):
            return e[i].mean(-2)
    elif stat == "rms":
        def f(e, i=slice(None)):
            return np.sqrt((e[i] ** 2).mean(-2))
    else:
        raise ValueError("stat must be 'mean' or 'rms'")
    r = np.stack([f(e_a[:, j:j + 1], idx)[:, 0] / f(e_b[:, j:j + 1], idx)[:, 0] for j in range(e_a.shape[1])], 1)
    point = f(e_a) / f(e_b)
    return point, np.percentile(r, 2.5, axis=0), np.percentile(r, 97.5, axis=0)


def wilcoxon_signed_rank(e_a: np.ndarray, e_b: np.ndarray) -> np.ndarray:
    """Two-sided Wilcoxon signed-rank p-values of the paired differences ``e_a - e_b`` per column.

    A distribution-free complement to the bootstrap: tests whether one model's errors are systematically smaller.
    """
    from scipy.stats import wilcoxon

    return np.array([float(wilcoxon(e_a[:, j], e_b[:, j]).pvalue) for j in range(e_a.shape[1])])


def regime_breakdown(points: np.ndarray, e_a: np.ndarray, e_b: np.ndarray, min_count: int = 20) -> list[dict]:
    """MAE of two models per angle-of-attack band and per Reynolds-number band."""
    rows = []
    for name, col, edges in (("alpha", 4, ALPHA_EDGES), ("Re", 3, RE_EDGES)):
        v = points[:, col]
        for lo, hi in pairwise(edges):
            k = (v >= lo) & (v < hi)
            if k.sum() < min_count:
                continue
            a, b = e_a[k].mean(0), e_b[k].mean(0)
            rows.append({"variable": name, "lo": float(lo), "hi": float(min(hi, 1e7) if name == "Re" else hi),
                         "n": int(k.sum()), "a": dict(zip(COLUMNS, map(float, a))),
                         "b": dict(zip(COLUMNS, map(float, b)))})
    return rows


def compare(pred: np.ndarray, others: dict[str, np.ndarray], y: np.ndarray, points: np.ndarray,
            n_boot: int = 2000) -> dict:
    """Full paired comparison of one model against reference models.

    Returns
    -------
    dict
        ``metrics[name]`` for every model (``"model"`` for ``pred``), and for every reference model
        ``vs[name]`` with MAE and RMSE ratios (+ 95 % CIs), Wilcoxon p-values, the fraction of points where the model
        is more accurate, and the regime breakdown.
    """
    res: dict = {"n_test": len(y), "metrics": {"model": error_metrics(pred, y)}, "vs": {}}
    ea = np.abs(pred - y)
    for name, pr in others.items():
        res["metrics"][name] = error_metrics(pr, y)
        eb = np.abs(pr - y)
        r, lo, hi = bootstrap_ratio(ea, eb, n_boot)
        rr, rlo, rhi = bootstrap_ratio(ea, eb, n_boot, stat="rms")
        res["vs"][name] = {
            "mae_ratio": {c: [float(r[j]), float(lo[j]), float(hi[j])] for j, c in enumerate(COLUMNS)},
            "rmse_ratio": {c: [float(rr[j]), float(rlo[j]), float(rhi[j])] for j, c in enumerate(COLUMNS)},
            "wilcoxon_p": dict(zip(COLUMNS, map(float, wilcoxon_signed_rank(ea, eb)))),
            "win_fraction": dict(zip(COLUMNS, map(float, (ea < eb).mean(0)))),
            "breakdown": regime_breakdown(points, ea, eb),
        }
    return res
