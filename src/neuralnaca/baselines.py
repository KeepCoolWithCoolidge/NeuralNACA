"""Reference model: NeuralFoil (Sharpe & Hansman, arXiv:2503.16323)."""

from __future__ import annotations

import math

import numpy as np

from .geometry import naca4_coordinates

__all__ = ["NEURALFOIL_SIZES", "NF_TRAIN_RUNS", "neuralfoil_predict", "neuralfoil_version"]

#: XFOIL runs *attempted* to build NeuralFoil's training set (arXiv:2503.16323, Sec. III.E); 56 % converged.
NF_TRAIN_RUNS = 7_913_292
NEURALFOIL_SIZES = ("xxsmall", "xsmall", "small", "medium", "large", "xlarge", "xxlarge", "xxxlarge")


def neuralfoil_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("neuralfoil")
    except Exception:
        return None


def neuralfoil_predict(points: np.ndarray, model_size: str = "xxxlarge", n_crit: float = 9.0) -> np.ndarray:
    """NeuralFoil predictions ``[CL, ln CD, CM]`` for conditions ``[m, p, t, Re, alpha]``.

    The exact NACA coordinates (the same ones handed to XFOIL) are passed to
    ``neuralfoil.get_aero_from_coordinates`` with free transition (``xtr = 1``), the given ``n_crit`` and Mach 0.
    NeuralFoil fits its own Kulfan parameterisation to them internally; that fitting step is part of NeuralFoil's
    pipeline and therefore part of its error.
    """
    import neuralfoil as nf

    if model_size not in NEURALFOIL_SIZES:
        raise ValueError(f"unknown NeuralFoil size {model_size!r}")
    points = np.asarray(points, dtype=float)
    out = np.zeros((len(points), 3))
    for i, (m, p, t, re, a) in enumerate(points):
        o = nf.get_aero_from_coordinates(naca4_coordinates(m, p, t), alpha=float(a), Re=float(re), n_crit=n_crit,
                                         model_size=model_size)
        out[i] = [float(np.ravel(o["CL"])[0]), math.log(float(np.ravel(o["CD"])[0])), float(np.ravel(o["CM"])[0])]
    return out
