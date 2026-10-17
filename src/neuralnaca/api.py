"""High-level prediction API."""

from __future__ import annotations

import functools
import hashlib
import warnings
from importlib import resources
from pathlib import Path

import numpy as np

from .domain import DEFAULT_DOMAIN, Domain, DomainWarning, UnsupportedWarning
from .geometry import parse_naca
from .model import FoilNet, load_model, predict_tensors, to_tensors
from .panel import airfoil_constants
from .support import SupportModel

__all__ = ["DEFAULT_WEIGHTS", "NeuralNACA", "get_aero"]

#: File name of the shipped weights (5-member ensemble trained on 4,096 XFOIL runs, benchmark seed 0).
DEFAULT_WEIGHTS = "naca4_k5_n4096_s0.npz"
_CACHE_LIMIT = 100_000                     # memoised panel solutions per model instance


class NeuralNACA:
    """A trained surrogate for NACA 4-digit polars (``CL``, ``CD``, ``CM``; ``n_crit = 9``, Mach 0, free transition).

    Use :meth:`pretrained` for the shipped model or :meth:`load` for your own weights file.

    Examples
    --------
    >>> from neuralnaca import NeuralNACA
    >>> model = NeuralNACA.pretrained()
    >>> aero = model.predict("2412", alpha=[0.0, 4.0], Re=1e6)
    >>> sorted(aero)
    ['CD', 'CD_std', 'CL', 'CL_std', 'CM', 'CM_std', 'in_domain', 'lnCD_std', 'supported', 'xfoil_convergence']
    """

    def __init__(self, net: FoilNet, meta: dict | None = None, domain: Domain = DEFAULT_DOMAIN,
                 source: str | None = None, support: SupportModel | None = None):
        self.net = net.eval()
        self.meta = meta or {}
        self.domain = domain
        self.source = source
        self.support = support
        self._const_cache: dict[tuple[float, float, float], np.ndarray] = {}

    # ---------------------------------------------------------------------------------------------- constructors
    @classmethod
    def load(cls, path: str | Path, domain: Domain = DEFAULT_DOMAIN, support: str | Path | None = "auto") -> NeuralNACA:
        """Load a weights file written by :func:`neuralnaca.model.save_model`.

        ``support="auto"`` also loads ``<name>.support.npz`` next to the weights if it exists (see
        :mod:`neuralnaca.support`); pass a path to use another file or ``None`` to skip.
        """
        net, meta = load_model(path)
        sup = None
        if support == "auto":
            cand = Path(path).with_suffix(".support.npz")
            sup = SupportModel.load(cand) if cand.exists() else None
        elif support is not None:
            sup = SupportModel.load(support)
        return cls(net, meta, domain, str(path), sup)

    @classmethod
    def pretrained(cls) -> NeuralNACA:
        """The shipped model (see :data:`DEFAULT_WEIGHTS`)."""
        ref = resources.files("neuralnaca") / "weights" / DEFAULT_WEIGHTS
        with resources.as_file(ref) as p:
            return cls.load(p)

    # ---------------------------------------------------------------------------------------------- info
    @property
    def ensemble_size(self) -> int:
        return self.net.K

    def fingerprint(self) -> str:
        """SHA-256 (first 16 hex digits) of the network parameters - identifies a model unambiguously."""
        h = hashlib.sha256()
        for p in self.net.parameters():
            h.update(p.detach().cpu().numpy().tobytes())
        return h.hexdigest()[:16]

    def __repr__(self) -> str:
        return (f"NeuralNACA(variant={self.net.variant!r}, ensemble={self.net.K}, "
                f"fingerprint={self.fingerprint()!r}, source={self.source!r})")

    # ---------------------------------------------------------------------------------------------- prediction
    def _constants(self, mpt: np.ndarray) -> np.ndarray:
        """Panel constants for each row of ``mpt``, computed once per distinct airfoil (memoised)."""
        uniq, inv = np.unique(mpt, axis=0, return_inverse=True)
        if len(self._const_cache) + len(uniq) > _CACHE_LIMIT:              # bound the memory of long sweeps
            self._const_cache.clear()
        missing = [tuple(r) for r in uniq if tuple(r) not in self._const_cache]
        if missing:
            new = airfoil_constants(np.array(missing))
            for k, c in zip(missing, new):
                self._const_cache[k] = c
        table = np.stack([self._const_cache[tuple(r)] for r in uniq])
        return table[np.ravel(inv)]

    def predict(self, airfoil: str | None = None, *, alpha, Re, m=None, p=None, t=None,
                on_domain_error: str = "warn", return_members: bool = False) -> dict[str, np.ndarray]:
        """Predict aerodynamic coefficients.

        Parameters
        ----------
        airfoil
            NACA 4-digit designation (``"2412"``, ``"NACA 0012"``). Alternatively give ``m, p, t``.
        alpha
            Angle of attack in degrees (scalar or array).
        Re
            Chord Reynolds number (scalar or array).
        m, p, t
            Maximum camber, camber position and thickness as chord fractions (scalars or arrays); used instead of
            ``airfoil`` for non-integer sections. ``p`` is ignored when ``m == 0``.
        on_domain_error
            What to do for conditions outside the validated domain: ``"warn"`` (default; emits a
            :class:`~neuralnaca.DomainWarning`), ``"raise"`` or ``"ignore"``. The ``in_domain`` output flags them
            either way.
        return_members
            Also return the individual ensemble members' predictions (``members``: ``(K, ..., 3)`` with columns
            ``CL, ln CD, CM``).

        Returns
        -------
        dict of numpy.ndarray
            ``CL``, ``CD``, ``CM`` (ensemble mean; ``CD = exp(mean ln CD)``), their ensemble standard deviations
            ``CL_std``, ``lnCD_std`` (relative drag spread), ``CD_std`` (``= CD * lnCD_std``), ``CM_std``,
            ``in_domain`` (inside the validated box), ``xfoil_convergence`` (estimated probability that XFOIL
            converges at this condition, from the training data) and ``supported`` (in the domain, convergence
            probability >= 0.5 and drag inside the training range). All arrays have the broadcast shape of the inputs
            (``()`` for scalar inputs). Without a support model, ``xfoil_convergence`` is NaN and ``supported``
            equals ``in_domain``.

        Notes
        -----
        The ensemble spread is a heuristic indicator of model disagreement, not a calibrated error bar. Predictions
        that are not ``supported`` are extrapolations into a region where XFOIL itself rarely converges (mostly
        deep stall of thin sections at low Reynolds number); they can be physically meaningless and are flagged with
        an :class:`~neuralnaca.domain.UnsupportedWarning`.
        """
        if on_domain_error not in ("warn", "raise", "ignore"):
            raise ValueError("on_domain_error must be 'warn', 'raise' or 'ignore'")
        if airfoil is not None:
            if any(v is not None for v in (m, p, t)):
                raise ValueError("give either `airfoil` or `m, p, t`, not both")
            m, p, t = parse_naca(airfoil)
        elif any(v is None for v in (m, p, t)):
            raise ValueError("give an `airfoil` designation or all of `m, p, t`")
        arrs = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (m, p, t, Re, alpha)))
        shape = arrs[0].shape
        flat = np.stack([a.ravel() for a in arrs], 1)
        if not np.isfinite(flat).all():
            raise ValueError("inputs must be finite")
        if (flat[:, 2] <= 0).any() or (flat[:, 3] <= 0).any():
            raise ValueError("thickness and Reynolds number must be positive")
        cambered = np.abs(flat[:, 0]) > 0
        if (cambered & ((flat[:, 1] <= 0) | (flat[:, 1] >= 1))).any():
            raise ValueError("a cambered section needs 0 < p < 1")
        flat[~cambered, 1] = 0.0                                     # p is irrelevant for symmetric sections
        inside = self.domain.contains(*flat.T)
        if not inside.all() and on_domain_error != "ignore":
            msg = (f"{(~inside).sum()} of {inside.size} conditions lie outside the validated domain "
                   f"({self.domain.describe()}); predictions there are extrapolations.")
            if on_domain_error == "raise":
                raise ValueError(msg)
            warnings.warn(msg, DomainWarning, stacklevel=2)
        members = predict_tensors(self.net, to_tensors(flat, self._constants(flat[:, :3])))   # (K, n, 3)
        mean, std = members.mean(0), members.std(0)
        cd = np.exp(mean[:, 1])
        if self.support is not None:
            p_conv = self.support.convergence_probability(flat)
            supported = inside & (p_conv >= self.support.threshold) & (cd <= self.support.cd_max)
        else:
            p_conv, supported = np.full(len(flat), np.nan), inside
        unsupported = inside & ~supported
        if unsupported.any() and on_domain_error != "ignore":
            msg = (f"{unsupported.sum()} of {inside.size} conditions lie in a region without training support "
                   "(XFOIL rarely converges there, or the predicted drag exceeds the training range); see the "
                   "'supported' output.")
            if on_domain_error == "raise":
                raise ValueError(msg)
            warnings.warn(msg, UnsupportedWarning, stacklevel=2)
        out = {
            "CL": mean[:, 0], "CD": cd, "CM": mean[:, 2],
            "CL_std": std[:, 0], "lnCD_std": std[:, 1], "CD_std": cd * std[:, 1], "CM_std": std[:, 2],
            "in_domain": inside, "xfoil_convergence": p_conv, "supported": supported,
        }
        res = {k: np.asarray(v, dtype=bool if k in ("in_domain", "supported") else float).reshape(shape)
               for k, v in out.items()}
        if return_members:
            res["members"] = members.reshape((self.net.K, *shape, 3))
        return res

    def polar(self, airfoil: str, Re: float, alpha=None, **kwargs) -> dict[str, np.ndarray]:
        """Convenience: a polar ``alpha -> (CL, CD, CM)`` at one Reynolds number.

        ``alpha`` defaults to the validated range in 0.5 deg steps. The returned dict also contains ``alpha``.
        """
        if alpha is None:
            alpha = np.arange(self.domain.alpha[0], self.domain.alpha[1] + 1e-9, 0.5)
        alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
        out = self.predict(airfoil, alpha=alpha, Re=Re, **kwargs)
        out["alpha"] = alpha
        return out


@functools.lru_cache(maxsize=1)
def _default_model() -> NeuralNACA:
    return NeuralNACA.pretrained()


def get_aero(airfoil: str | None = None, *, alpha, Re, m=None, p=None, t=None,
             on_domain_error: str = "warn") -> dict[str, np.ndarray]:
    """Predict with the shipped model (loaded once and cached). See :meth:`NeuralNACA.predict`.

    Examples
    --------
    >>> import neuralnaca
    >>> aero = neuralnaca.get_aero("NACA 4412", alpha=5.0, Re=1e6)
    >>> 0.9 < float(aero["CL"]) < 1.1
    True
    """
    return _default_model().predict(airfoil, alpha=alpha, Re=Re, m=m, p=p, t=t, on_domain_error=on_domain_error)
