"""Where is a prediction backed by training data?

XFOIL does not converge everywhere in the domain (deep stall of thin sections at low Reynolds number is the main
region). The surrogate never saw data there, so its predictions are unconstrained extrapolations - drag in particular
can grow without bound. The benchmark cannot score these points (XFOIL has no answer either), so they must be
*flagged* instead.

:class:`SupportModel` estimates the probability that XFOIL converges at a condition from the training set itself
(a distance-weighted k-nearest-neighbour vote over converged / non-converged training runs; no extra XFOIL runs), and
also checks that the predicted drag stays inside the range of the training targets. A prediction is *supported* if
it is inside the domain, the estimated convergence probability is at least ``threshold`` and ``CD`` does not exceed
``cd_max``. The documentation reports how well this flag separates converged from non-converged held-out points.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .domain import DEFAULT_DOMAIN, Domain

__all__ = ["SupportModel", "support_features"]

SUPPORT_FORMAT = "neuralnaca-support/1"


def support_features(points: np.ndarray, domain: Domain = DEFAULT_DOMAIN) -> np.ndarray:
    """Inputs scaled to the unit box (camber position gated exactly as in the network features)."""
    m, p, t, re, a = np.asarray(points, dtype=float).T
    pe = 0.45 + (p - 0.45) * (1.0 - np.exp(-(m / 0.01) ** 2))
    d = domain
    return np.stack([(m - d.m[0]) / (d.m[1] - d.m[0]), (pe - d.p[0]) / (d.p[1] - d.p[0]),
                     (t - d.t[0]) / (d.t[1] - d.t[0]),
                     (np.log(re) - np.log(d.re[0])) / (np.log(d.re[1]) - np.log(d.re[0])),
                     (a - d.alpha[0]) / (d.alpha[1] - d.alpha[0])], 1)


@dataclass
class SupportModel:
    """k-nearest-neighbour estimate of XFOIL convergence plus an output-range guard.

    Attributes
    ----------
    points
        ``(n, 5)`` training conditions (converged *and* non-converged).
    converged
        ``(n,)`` whether XFOIL converged at each training condition.
    cd_max
        Largest drag coefficient among the converged training points.
    k, threshold, eps
        Neighbours, decision threshold on the convergence probability, distance regulariser of the weights.
    """

    points: np.ndarray
    converged: np.ndarray
    cd_max: float
    k: int = 10
    threshold: float = 0.5
    eps: float = 0.02

    def __post_init__(self):
        from scipy.spatial import cKDTree

        self.points = np.asarray(self.points, dtype=float)
        self.converged = np.asarray(self.converged, dtype=bool)
        if len(self.points) != len(self.converged) or len(self.points) < self.k:
            raise ValueError("need at least k training points with a convergence flag each")
        self._tree = cKDTree(support_features(self.points))

    @classmethod
    def from_dataset(cls, ds, **kwargs) -> SupportModel:
        """Build from an :class:`~neuralnaca.data.XfoilDataset` (the model's training set)."""
        ok = ds.converged
        return cls(ds.points, ok, float(ds.results[ok, 1].max()), **kwargs)

    def convergence_probability(self, points: np.ndarray) -> np.ndarray:
        """Distance-weighted fraction of converged runs among the ``k`` nearest training conditions."""
        d, idx = self._tree.query(support_features(points), k=self.k)
        w = 1.0 / (d + self.eps)
        return (self.converged[idx] * w).sum(1) / w.sum(1)

    def supported(self, points: np.ndarray, cd: np.ndarray, domain: Domain = DEFAULT_DOMAIN) -> np.ndarray:
        p = np.asarray(points, dtype=float)
        return (domain.contains(*p.T) & (self.convergence_probability(p) >= self.threshold)
                & (np.asarray(cd) <= self.cd_max))

    def save(self, path: str | Path) -> None:
        header = {"format": SUPPORT_FORMAT, "cd_max": self.cd_max, "k": self.k, "threshold": self.threshold,
                  "eps": self.eps}
        with open(path, "wb") as fh:
            np.savez(fh, header=np.array(json.dumps(header)), points=self.points, converged=self.converged)

    @classmethod
    def load(cls, path: str | Path) -> SupportModel:
        with np.load(path, allow_pickle=False) as z:
            h = json.loads(str(z["header"]))
            if h.get("format") != SUPPORT_FORMAT:
                raise ValueError(f"{path} is not a NeuralNACA support file")
            return cls(z["points"], z["converged"], h["cd_max"], h["k"], h["threshold"], h["eps"])
