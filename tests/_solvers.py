"""Stand-ins for XFOIL used by the tests (module level so that spawned worker processes can import them)."""

from __future__ import annotations

import math
import os


def analytic_teacher(pt, cfg):
    """A smooth, physically flavoured synthetic 'XFOIL': thin-airfoil lift, a drag bucket and a camber moment."""
    m, p, t, re, alpha = pt
    a = math.radians(alpha)
    cl = 2 * math.pi * (1 + 0.77 * t) * math.sin(a) + 4 * math.pi * m * (1 - 0.3 * (p - 0.4)) - 0.02 * alpha ** 2 / 15
    cd = 0.074 / (re ** 0.2) * (1 + 2 * t) + 0.01 * (cl - 0.3) ** 2 * 0.1 + 0.0002 * abs(alpha)
    cm = -math.pi * m * (1.0 + 0.5 * (p - 0.4)) + 0.002 * alpha * t
    return cl, cd, cm


def crashing_teacher(pt, cfg):
    """Kills its worker process (like a segmentation fault inside XFOIL) for two 'poison' angles."""
    if pt[4] in (13.0, 7.0):
        os._exit(7)
    return analytic_teacher(pt, cfg)


def raising_teacher(pt, cfg):
    """Raises for negative angles (an ordinary Python exception inside a worker)."""
    if pt[4] < 0:
        raise RuntimeError("boom")
    return analytic_teacher(pt, cfg)


def nonconverging_teacher(pt, cfg):
    """Returns NaN / non-positive drag like an unconverged XFOIL point for alpha > 10."""
    if pt[4] > 10:
        return (math.nan, math.nan, math.nan) if pt[4] > 12 else (0.5, -1.0, 0.0)
    return analytic_teacher(pt, cfg)
