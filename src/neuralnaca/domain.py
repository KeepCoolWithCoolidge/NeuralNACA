"""The validated input domain and the experimental designs used to sample it.

Every number this package reports refers to the domain defined here. Predictions outside of it are
extrapolations and are flagged with a :class:`DomainWarning`.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass

import numpy as np

__all__ = ["DEFAULT_DOMAIN", "Domain", "DomainWarning", "UnsupportedWarning", "sample_points"]


class DomainWarning(UserWarning):
    """Issued when a prediction is requested outside the domain the model was trained and validated on."""


class UnsupportedWarning(DomainWarning):
    """Issued for conditions inside the domain where the model has no training support (XFOIL itself rarely
    converges there) or where the predicted drag exceeds the range of the training data."""


@dataclass(frozen=True)
class Domain:
    """Box-shaped input domain of the surrogate.

    Attributes
    ----------
    m, p, t
        Ranges of NACA 4-digit maximum camber, camber position and thickness (fractions of chord).
        ``p`` is irrelevant (and not checked) for symmetric sections (``m == 0``).
    re
        Chord Reynolds number range (sampled log-uniformly).
    alpha
        Angle-of-attack range in degrees.
    sym_fraction
        Fraction of sampled airfoils that are forced to be symmetric (``m = 0``), so that the
        symmetric sections - a set of measure zero in a continuous design - are well represented.
    """

    m: tuple[float, float] = (0.0, 0.08)
    p: tuple[float, float] = (0.2, 0.7)
    t: tuple[float, float] = (0.06, 0.18)
    re: tuple[float, float] = (1e5, 1e7)
    alpha: tuple[float, float] = (-5.0, 15.0)
    sym_fraction: float = 0.20

    def contains(self, m, p, t, re, alpha, atol: float = 1e-9) -> np.ndarray:
        """Element-wise membership test (inputs broadcast against each other).

        Returns
        -------
        numpy.ndarray of bool
            ``True`` where the condition lies inside the domain.
        """
        m, p, t, re, alpha = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (m, p, t, re, alpha)))

        def inside(v, rng):
            return (v >= rng[0] - atol) & (v <= rng[1] + atol)

        p_ok = (np.abs(m) <= atol) | inside(p, self.p)
        rel = 1e-9
        re_ok = (re >= self.re[0] * (1 - rel)) & (re <= self.re[1] * (1 + rel))
        return inside(m, self.m) & p_ok & inside(t, self.t) & re_ok & inside(alpha, self.alpha)

    def to_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        return (f"m in [{self.m[0]:g}, {self.m[1]:g}], p in [{self.p[0]:g}, {self.p[1]:g}], "
                f"t in [{self.t[0]:g}, {self.t[1]:g}], Re in [{self.re[0]:.0e}, {self.re[1]:.0e}], "
                f"alpha in [{self.alpha[0]:g}, {self.alpha[1]:g}] deg")


DEFAULT_DOMAIN = Domain()


def sample_points(n: int, kind: str = "sobol", seed: int = 0, domain: Domain = DEFAULT_DOMAIN) -> np.ndarray:
    """Sample flow conditions ``[m, p, t, Re, alpha]`` over a domain.

    Parameters
    ----------
    n
        Number of points.
    kind
        ``"sobol"`` - scrambled Sobol sequence (space filling; used for *training* sets; powers of two keep its
        balance properties and every prefix of length ``2**k`` is itself a balanced design), or
        ``"random"`` - i.i.d. uniform (used for *test* sets, so that test error is an unbiased estimate of the
        domain-averaged error).
    seed
        Seed of the scrambling / random generator. Training uses ``seed``, testing ``seed + 1000``, so the two
        sets never share a random stream.
    domain
        Domain to sample.

    Returns
    -------
    numpy.ndarray, shape (n, 5)
        Columns ``m, p, t, Re, alpha``. Angles are rounded to 1e-3 deg (XFOIL reads them with limited precision).
        A fraction ``domain.sym_fraction`` of the rows has ``m = 0``.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if kind == "sobol":
        from scipy.stats import qmc

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")          # "balance properties" warning for non powers of two
            u = qmc.Sobol(d=6, scramble=True, seed=seed).random(n)
    elif kind == "random":
        u = np.random.default_rng(seed).random((n, 6))
    else:
        raise ValueError(f"unknown design kind {kind!r}; use 'sobol' or 'random'")
    d = domain
    m = d.m[0] + u[:, 0] * (d.m[1] - d.m[0])
    m = np.where(u[:, 5] < d.sym_fraction, 0.0, m)
    p = d.p[0] + u[:, 1] * (d.p[1] - d.p[0])
    t = d.t[0] + u[:, 2] * (d.t[1] - d.t[0])
    lr = math.log(d.re[0]) + u[:, 3] * (math.log(d.re[1]) - math.log(d.re[0]))
    a = np.round(d.alpha[0] + u[:, 4] * (d.alpha[1] - d.alpha[0]), 3)
    return np.stack([m, p, t, np.exp(lr), a], 1)
