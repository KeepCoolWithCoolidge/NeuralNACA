r"""Network architecture, input features and the weights file format.

The model predicts ``y = [CL, ln CD, CM]`` as a *physics base* plus a learned correction:

.. math::

    y = y_\mathrm{base}(\text{panel method, flat-plate drag}) + \Delta y_\theta(\text{features})

and the correction is made exactly mirror-symmetric: reflecting a section about its chord line
(:math:`m\to-m`, :math:`\alpha\to-\alpha`) must give :math:`C_L\to-C_L`, :math:`C_M\to-C_M` and the same
:math:`C_D`. With :math:`x` the features and :math:`\bar x` the features of the reflected case,

.. math::

    \Delta C_L = \tfrac12[f_L(x) - f_L(\bar x)],\quad
    \Delta \ln C_D = \tfrac12[f_D(x) + f_D(\bar x)],\quad
    \Delta C_M = \tfrac12[f_M(x) - f_M(\bar x)].

This only holds if *every* feature transforms correctly under the reflection, which is why the camber-position
gate is an even function of ``m`` (see :data:`FEATURES_VERSION`).

Ablation variants (used in the benchmark to measure what each piece of physics contributes) remove one
ingredient at a time; ``"full"`` is the shipped model.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .domain import DEFAULT_DOMAIN, Domain
from .panel import airfoil_constants, inviscid

__all__ = [
    "FEATURES_VERSION",
    "VARIANTS",
    "EnsembleMLP",
    "FoilNet",
    "IncompatibleWeightsError",
    "VariantSpec",
    "check_reflection",
    "drag_prior",
    "load_model",
    "make_inputs",
    "predict_tensors",
    "save_model",
    "to_tensors",
]

#: Version of the feature definition in :func:`make_inputs`. Weights files record it and loading a file with a
#: different version is refused. v1 (pre-release) used the *odd* gate ``tanh(m / 0.01)`` for the camber position,
#: which silently broke the exact reflection symmetry for negative camber; v2 uses the even gate
#: ``1 - exp(-(m / 0.01)^2)``.
FEATURES_VERSION = 2
WEIGHTS_FORMAT = "neuralnaca-weights/1"

RE0 = math.log(1e6)
LN_CD0 = math.log(0.01)                                      # constant drag "prior" for variants without physics
LOSS_W = torch.tensor([1 / 0.01, 1 / 0.03, 1 / 0.002])      # ~ inverse of the accuracy we care about: CL, lnCD, CM
OUT_SCALE = torch.tensor([0.1, 1.0, 0.01])                  # natural scale of the network's raw outputs


@dataclass(frozen=True)
class VariantSpec:
    """Which physics ingredients a model variant uses."""

    panel: bool          #: inviscid panel-method CL/CM as base *and* as input features
    symmetric: bool      #: exact mirror symmetry built into the output
    drag_prior: bool     #: flat-plate skin-friction prior as the ln(CD) base (else a constant)
    gate: bool           #: camber position switched off for uncambered sections


VARIANTS: dict[str, VariantSpec] = {
    "full": VariantSpec(panel=True, symmetric=True, drag_prior=True, gate=True),
    "no_symmetry": VariantSpec(panel=True, symmetric=False, drag_prior=True, gate=True),
    "no_panel": VariantSpec(panel=False, symmetric=True, drag_prior=True, gate=True),
    "no_drag_prior": VariantSpec(panel=True, symmetric=True, drag_prior=False, gate=True),
    "plain": VariantSpec(panel=False, symmetric=False, drag_prior=False, gate=False),
}


class IncompatibleWeightsError(ValueError):
    """The weights file was written by an incompatible version of the feature definition or file format."""


def drag_prior(lnre: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """ln of a flat-plate turbulent skin-friction drag estimate with a thickness form factor.

    ``CD ~ 2 Cf (1 + 2 t + 60 t^4)`` with the Prandtl-Schlichting ``Cf = 0.455 / (log10 Re)^2.58``.
    """
    cf = 0.455 / (lnre / math.log(10.0)) ** 2.58
    return torch.log(2.0 * cf * (1.0 + 2.0 * t + 60.0 * t ** 4))


def _spec(variant: str) -> VariantSpec:
    try:
        return VARIANTS[variant]
    except KeyError:
        raise ValueError(f"unknown model variant {variant!r}; choose from {sorted(VARIANTS)}") from None


def n_features(variant: str = "full") -> int:
    return 7 if _spec(variant).panel else 5


def make_inputs(m, p, t, lnre, alpha, const, variant: str = "full"):
    """Network features, the features of the reflected case and the physics base.

    All arguments are tensors of identical shape (``const`` has a trailing dimension of 6).

    Returns
    -------
    x : torch.Tensor (..., n_features)
    xm : torch.Tensor or None
        Features of the reflected case (``m -> -m``, ``alpha -> -alpha``, inviscid ``CL, CM -> -CL, -CM``);
        ``None`` for non-symmetric variants.
    base : torch.Tensor (..., 3)
        ``[CL_inv, ln CD_prior, CM_inv]`` (zeros / a constant where the variant has no physics).
    """
    spec = _spec(variant)
    if spec.gate:
        # Camber position is irrelevant without camber. The gate MUST be even in m: reflecting a section leaves p
        # unchanged, so the feature must be unchanged too (an odd gate mirrors pe about 0.45 for negative camber).
        pe = 0.45 + (p - 0.45) * (1.0 - torch.exp(-(m / 0.01) ** 2))
    else:
        pe = p
    if spec.panel:
        cl_i, cm_i = inviscid(const, alpha)
    else:
        cl_i = cm_i = torch.zeros_like(alpha)
    ln_cd0 = drag_prior(lnre, t) if spec.drag_prior else torch.full_like(lnre, LN_CD0)

    def mk(sg):
        cols = [sg * m / 0.04, (pe - 0.45) / 0.25, (t - 0.12) / 0.06, (lnre - RE0) / 1.2, sg * alpha / 10.0]
        if spec.panel:
            cols += [sg * cl_i, sg * cm_i * 10.0]
        return torch.stack(cols, -1)

    xm = mk(-1.0) if spec.symmetric else None
    return mk(1.0), xm, torch.stack([cl_i, ln_cd0, cm_i], -1)


class EnsembleMLP(nn.Module):
    """``K`` independent MLPs evaluated together with batched matrix products: ``(K, B, d) -> (K, B, dout)``.

    SiLU activations; PyTorch-default (fan-in uniform) initialisation, drawn member by member so that the members
    are independent.
    """

    def __init__(self, K: int, din: int, dout: int, width: int = 128, depth: int = 4):
        super().__init__()
        dims = [din] + [width] * depth + [dout]
        self.W, self.b = nn.ParameterList(), nn.ParameterList()
        for i in range(len(dims) - 1):
            bound = 1.0 / math.sqrt(dims[i])
            self.W.append(nn.Parameter(torch.empty(K, dims[i], dims[i + 1]).uniform_(-bound, bound)))
            self.b.append(nn.Parameter(torch.empty(K, 1, dims[i + 1]).uniform_(-bound, bound)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = torch.baddbmm(b, x, W)
            if i < len(self.W) - 1:
                x = nn.functional.silu(x)
        return x


class FoilNet(nn.Module):
    """Physics base + (optionally mirror-symmetrised) ensemble correction.

    ``forward`` returns ``(K, B, 3)`` per-member predictions of ``[CL, ln CD, CM]``.
    """

    def __init__(self, K: int, din: int, width: int = 128, depth: int = 4, variant: str = "full"):
        super().__init__()
        if din != n_features(variant):
            raise ValueError(f"variant {variant!r} expects {n_features(variant)} input features, got {din}")
        self.f = EnsembleMLP(K, din, 3, width, depth)
        self.K, self.din, self.width, self.depth, self.variant = K, din, width, depth, variant

    @property
    def symmetric(self) -> bool:
        return _spec(self.variant).symmetric

    def forward(self, x: torch.Tensor, xm: torch.Tensor | None, base: torch.Tensor) -> torch.Tensor:
        y = self.f(x) * OUT_SCALE
        if self.symmetric:
            if xm is None:
                raise ValueError("a symmetric model needs the reflected features xm")
            ym = self.f(xm) * OUT_SCALE
            y = torch.stack([0.5 * (y[..., 0] - ym[..., 0]), 0.5 * (y[..., 1] + ym[..., 1]),
                             0.5 * (y[..., 2] - ym[..., 2])], -1)
        return base + y


def to_tensors(points: np.ndarray, const: np.ndarray, results: np.ndarray | None = None) -> dict[str, torch.Tensor]:
    """Pack ``[m, p, t, Re, alpha]`` points, their panel constants and (optionally) XFOIL results as float32 tensors.

    ``results`` columns are ``[CL, CD, CM]``; the target stored under ``"y"`` is ``[CL, ln CD, CM]``.
    """
    points = np.asarray(points, dtype=float)

    def f(v):
        return torch.tensor(np.asarray(v), dtype=torch.float32)

    d = {"m": f(points[:, 0]), "p": f(points[:, 1]), "t": f(points[:, 2]), "lnre": f(np.log(points[:, 3])),
         "alpha": f(points[:, 4]), "const": f(const)}
    if results is not None:
        d["y"] = f(np.stack([results[:, 0], np.log(results[:, 1]), results[:, 2]], 1))
    return d


@torch.no_grad()
def predict_tensors(net: FoilNet, data: dict[str, torch.Tensor]) -> np.ndarray:
    """Per-member predictions ``(K, n, 3)`` of ``[CL, ln CD, CM]`` for packed inputs (see :func:`to_tensors`)."""
    d = {k: v[None].expand(net.K, *v.shape) for k, v in data.items() if k != "y"}
    x, xm, base = make_inputs(d["m"], d["p"], d["t"], d["lnre"], d["alpha"], d["const"], net.variant)
    return net(x, xm, base).numpy()


# ------------------------------------------------------------------------------------------------ weights files
def save_model(net: FoilNet, path: str | Path, meta: dict | None = None) -> None:
    """Write weights + metadata to a NumPy ``.npz`` file (no pickle; safe to load from untrusted sources)."""
    header = {
        "format": WEIGHTS_FORMAT, "features_version": FEATURES_VERSION, "variant": net.variant,
        "K": net.K, "din": net.din, "width": net.width, "depth": net.depth, "meta": meta or {},
    }
    arrays = {f"W{i}": W.detach().cpu().numpy() for i, W in enumerate(net.f.W)}
    arrays |= {f"b{i}": b.detach().cpu().numpy() for i, b in enumerate(net.f.b)}
    path = Path(path)
    with open(path, "wb") as fh:                         # explicit handle: np.savez would append ".npz" to the name
        np.savez(fh, header=np.array(json.dumps(header, sort_keys=True, default=_json_default)), **arrays)


def load_model(path: str | Path) -> tuple[FoilNet, dict]:
    """Load a weights file written by :func:`save_model`.

    Returns
    -------
    (net, meta)
        The network in eval mode and the free-form metadata stored with it.

    Raises
    ------
    IncompatibleWeightsError
        If the file has another format or feature version.
    """
    with np.load(path, allow_pickle=False) as z:
        if "header" not in z.files:
            raise IncompatibleWeightsError(f"{path} is not a NeuralNACA weights file (no header)")
        header = json.loads(str(z["header"]))
        if header.get("format") != WEIGHTS_FORMAT:
            raise IncompatibleWeightsError(f"{path}: unsupported weights format {header.get('format')!r}")
        if header.get("features_version") != FEATURES_VERSION:
            raise IncompatibleWeightsError(
                f"{path} was trained with feature definition v{header.get('features_version')}, this version of "
                f"neuralnaca needs v{FEATURES_VERSION}; retrain the model.")
        net = FoilNet(header["K"], header["din"], header["width"], header["depth"], header["variant"])
        n_layers = len(net.f.W)
        state = {f"f.W.{i}": torch.from_numpy(z[f"W{i}"].copy()) for i in range(n_layers)}
        state |= {f"f.b.{i}": torch.from_numpy(z[f"b{i}"].copy()) for i in range(n_layers)}
    net.load_state_dict(state)
    return net.eval(), header["meta"]


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


# ------------------------------------------------------------------------------------------------ symmetry check
@torch.no_grad()
def check_reflection(net: FoilNet, n: int = 400, seed: int = 12345, tol: float = 1e-4,
                     domain: Domain = DEFAULT_DOMAIN) -> dict:
    """Explicit test of the mirror symmetry on signed-camber geometries across the whole domain.

    The reflected section (``m -> -m``, same ``p, t, Re``; ``alpha -> -alpha``) gets its *own* panel solution, and the
    model must return ``CL' = -CL``, ``CM' = -CM``, ``ln CD' = ln CD``.

    Returns
    -------
    dict
        Maximum absolute violation per output (``max_CL``, ``max_lnCD``, ``max_CM``) and ``ok`` (all ``<= tol``).
        Non-symmetric ablation variants are expected to fail.
    """
    rng, D = np.random.default_rng(seed), domain
    m = rng.uniform(*D.m, n)
    m[: n // 8] = rng.uniform(0.0, 0.01, n // 8)                      # include the small-camber gate region
    cols = [m, rng.uniform(*D.p, n), rng.uniform(*D.t, n),
            np.exp(rng.uniform(np.log(D.re[0]), np.log(D.re[1]), n)), rng.uniform(*D.alpha, n)]
    P = np.stack(cols, 1)
    R = P * np.array([-1.0, 1.0, 1.0, 1.0, -1.0])
    yP = predict_tensors(net, to_tensors(P, airfoil_constants(P[:, :3]))).mean(0)
    yR = predict_tensors(net, to_tensors(R, airfoil_constants(R[:, :3]))).mean(0)
    worst = np.array([np.abs(yP[:, 0] + yR[:, 0]).max(), np.abs(yP[:, 1] - yR[:, 1]).max(),
                      np.abs(yP[:, 2] + yR[:, 2]).max()])
    return {"max_CL": float(worst[0]), "max_lnCD": float(worst[1]), "max_CM": float(worst[2]), "n": n, "tol": tol,
            "ok": bool(worst.max() <= tol)}
