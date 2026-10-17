"""NACA 4-digit section geometry.

The thickness distribution is the classical open-trailing-edge one (coefficient -0.1015), which is what XFOIL's
``NACA`` command and AeroSandbox's ``Airfoil("naca2412")`` generate. Thickness is applied perpendicular to the mean
camber line.
"""

from __future__ import annotations

import math
import re

import numpy as np
import torch

__all__ = ["format_naca", "naca4_coordinates", "naca4_nodes", "parse_naca"]

_CODE = re.compile(r"^(?:NACA)?[\s\-_]*(\d{4})$")


def parse_naca(code: str) -> tuple[float, float, float]:
    """Parse a NACA 4-digit designation.

    Accepts ``"2412"``, ``"NACA 2412"``, ``"naca2412"``, ``"NACA-2412"``.

    Returns
    -------
    (m, p, t)
        Maximum camber, its chordwise position and the maximum thickness, as fractions of chord.

    Raises
    ------
    ValueError
        If the string is not a 4-digit code, if a cambered section has no camber position (e.g. ``"2012"``), or if
        the thickness is zero.
    """
    match = _CODE.match(code.strip().upper())
    if match is None:
        raise ValueError(f"{code!r} is not a NACA 4-digit designation such as '2412'")
    d = match.group(1)
    m, p, t = int(d[0]) / 100.0, int(d[1]) / 10.0, int(d[2:]) / 100.0
    if m > 0 and p == 0:
        raise ValueError(f"NACA {d}: a cambered section needs a non-zero camber position (second digit)")
    if t == 0:
        raise ValueError(f"NACA {d}: thickness must be positive")
    if m == 0:
        p = 0.0
    return m, p, t


def format_naca(m: float, p: float, t: float) -> str:
    """Inverse of :func:`parse_naca` for parameters that correspond to an integer designation."""
    dm, dp, dt = round(m * 100), round(p * 10), round(t * 100)
    if not (math.isclose(dm, m * 100, abs_tol=1e-6) and math.isclose(dt, t * 100, abs_tol=1e-6)
            and (dm == 0 or math.isclose(dp, p * 10, abs_tol=1e-6))):
        raise ValueError(f"(m={m}, p={p}, t={t}) is not an integer NACA 4-digit designation")
    if not (0 <= dm <= 9 and 0 <= dp <= 9 and 1 <= dt <= 99):
        raise ValueError(f"(m={m}, p={p}, t={t}) is outside the NACA 4-digit range")
    return f"{dm}{dp if dm else 0}{dt:02d}"


def naca4_coordinates(m: float, p: float, t: float, n_per_side: int = 100) -> np.ndarray:
    """Coordinates in Selig order (upper trailing edge -> leading edge -> lower trailing edge).

    Cosine spacing with ``n_per_side`` points per surface (the leading-edge point is shared), i.e.
    ``2 * n_per_side - 1`` points in total. These are the coordinates handed to XFOIL (which re-panels them) and to
    NeuralFoil.

    Negative ``m`` is allowed and gives the section reflected about the chord line.
    """
    beta = np.linspace(0.0, np.pi, n_per_side)
    x = 0.5 * (1.0 - np.cos(beta))
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1015 * x ** 4)
    if abs(m) < 1e-9:
        yc, dyc = np.zeros_like(x), np.zeros_like(x)
    else:
        if not 0.0 < p < 1.0:
            raise ValueError("camber position p must lie in (0, 1) for a cambered section")
        yc = np.where(x < p, m / p ** 2 * (2 * p * x - x ** 2), m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))
        dyc = np.where(x < p, 2 * m / p ** 2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    th = np.arctan(dyc)
    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)
    return np.stack([np.concatenate([xu[::-1], xl[1:]]), np.concatenate([yu[::-1], yl[1:]])], 1)


def naca4_nodes(m: torch.Tensor, p: torch.Tensor, t: torch.Tensor, n_half: int = 60) -> torch.Tensor:
    """Batched, differentiable panel nodes for the panel solver.

    Clockwise ordering (lower trailing edge -> leading edge -> upper trailing edge), cosine spacing,
    ``n_half`` panels per surface.

    Parameters
    ----------
    m, p, t
        Tensors of shape ``(B,)`` (any floating dtype; the result has the same dtype).

    Returns
    -------
    torch.Tensor, shape (B, 2 * n_half + 1, 2)
    """
    beta = torch.linspace(0.0, math.pi, n_half + 1, dtype=m.dtype)
    x = (0.5 * (1.0 - torch.cos(beta)))[None, :].expand(m.shape[0], -1)
    mm, pp, tt = m[:, None], p[:, None].clamp(min=1e-3), t[:, None]
    yt = 5.0 * tt * (0.2969 * torch.sqrt(x.clamp(min=1e-12)) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3
                     - 0.1015 * x ** 4)
    fore = x < pp
    yc = torch.where(fore, mm / pp ** 2 * (2 * pp * x - x ** 2),
                     mm / (1 - pp) ** 2 * ((1 - 2 * pp) + 2 * pp * x - x ** 2))
    dyc = torch.where(fore, 2 * mm / pp ** 2 * (pp - x), 2 * mm / (1 - pp) ** 2 * (pp - x))
    th = torch.atan(dyc)
    xu, yu = x - yt * torch.sin(th), yc + yt * torch.cos(th)
    xl, yl = x + yt * torch.sin(th), yc - yt * torch.cos(th)
    return torch.stack([torch.cat([xl.flip(1), xu[:, 1:]], 1), torch.cat([yl.flip(1), yu[:, 1:]], 1)], -1)
