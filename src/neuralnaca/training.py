"""Training of the ensemble surrogate.

Recipe (all members trained simultaneously with batched matrix products, each on its *own* minibatch stream):

* loss: smooth-L1 (Huber, ``beta = 0.1``) on ``[CL, ln CD, CM]`` scaled by ``LOSS_W`` - robust to the occasional
  erratic XFOIL point near stall;
* Adam with a one-cycle learning-rate schedule (peak ``3e-3``);
* exponential moving average (EMA) of the weights over the second half of training - the EMA weights are the model;
* optional autograd "PINN" residuals on unlabelled collocation points (off by default; in our experiments they do not
  improve accuracy, see the documentation).
"""

from __future__ import annotations

import copy
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .model import LOSS_W, FoilNet, make_inputs
from .panel import inviscid

__all__ = ["RESIDUAL_BANDS", "TrainConfig", "physics_residuals", "train_ensemble"]

log = logging.getLogger(__name__)

#: Bands for the optional physics residuals; each was checked against XFOIL polar sweeps (fraction of XFOIL intervals
#: satisfying the band given in :func:`physics_residuals`).
RESIDUAL_BANDS = {"r_lo": 0.5, "r_hi": 1.3, "cl_pre": 0.8, "curv_max": 0.15, "dre_max": 0.1}


@dataclass(frozen=True)
class TrainConfig:
    """Hyper-parameters. The defaults are the ones used for every reported result."""

    ensemble: int = 5
    steps: int = 18000
    batch_size: int = 512
    lr: float = 3e-3
    huber_beta: float = 0.1
    ema: float = 0.9995
    width: int = 128
    depth: int = 4
    seed: int = 0
    variant: str = "full"
    residuals: bool = False
    residual_weight: float = 1.0
    n_collocation_batch: int = 128

    def to_dict(self) -> dict:
        return asdict(self)


def physics_residuals(net: FoilNet, col: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Autograd residuals on unlabelled collocation points (all tensors shaped ``(K, nc)``).

    * ``slope``: pre-stall ``dCL/dalpha`` within ``[0.5, 1.3]`` x the inviscid slope (XFOIL: 97 % of intervals);
    * ``curv``: ``|d2CL/dalpha2| <= 0.15`` per deg^2 in the pre-stall range (XFOIL: 99 %);
    * ``re``: ``d ln(CD) / d ln(Re) <= 0.1`` - profile drag falls with Reynolds number (XFOIL: 98.7 %).
    """
    alpha = col["alpha"].clone().requires_grad_(True)
    lnre = col["lnre"].clone().requires_grad_(True)
    x, xm, base = make_inputs(col["m"], col["p"], col["t"], lnre, alpha, col["const"], net.variant)
    y = net(x, xm, base)
    dcl = torch.autograd.grad(y[..., 0].sum(), alpha, create_graph=True)[0]
    d2cl = torch.autograd.grad(dcl.sum(), alpha, create_graph=True)[0]
    dre = torch.autograd.grad(y[..., 1].sum(), lnre, create_graph=True)[0]
    cl_i, _ = inviscid(col["const"], alpha)
    s_inv = torch.autograd.grad(cl_i.sum(), alpha, retain_graph=True)[0]
    b = RESIDUAL_BANDS
    ratio = dcl / s_inv.clamp(min=1e-3)
    pre = torch.sigmoid((b["cl_pre"] - cl_i.abs()) / 0.08).detach()
    return {"slope": (pre * (torch.relu(b["r_lo"] - ratio) + torch.relu(ratio - b["r_hi"]))).mean(),
            "curv": (pre * (torch.relu(d2cl.abs() - b["curv_max"]) / b["curv_max"]) ** 2).mean(),
            "re": torch.relu(dre - b["dre_max"]).mean()}


def train_ensemble(data: dict[str, torch.Tensor], cfg: TrainConfig | None = None,
                   col: dict[str, torch.Tensor] | None = None, tag: str = "train",
                   callback: Callable[[int, float], None] | None = None) -> FoilNet:
    """Train an ensemble on packed data (see :func:`neuralnaca.model.to_tensors`).

    Parameters
    ----------
    data
        Packed *converged* training points including the target ``"y"``.
    cfg
        Hyper-parameters (defaults: :class:`TrainConfig`).
    col
        Packed unlabelled collocation points; required if ``cfg.residuals``.
    tag
        Label for progress logging.
    callback
        Called as ``callback(step, loss)`` after every step (e.g. for tests or custom progress bars).

    Returns
    -------
    FoilNet
        The EMA network in eval mode.

    Notes
    -----
    Training is deterministic for a given seed, PyTorch build and machine. Across machines / BLAS builds the
    floating-point summation order can differ and runs diverge slightly; the documentation quantifies the
    resulting spread of the test metrics.
    """
    cfg = cfg or TrainConfig()
    if cfg.steps < 10:
        raise ValueError("train for at least 10 steps")
    if cfg.residuals and col is None:
        raise ValueError("cfg.residuals=True needs collocation points `col`")
    torch.manual_seed(cfg.seed)
    n = data["m"].shape[0]
    K = cfg.ensemble
    x0, _, _ = make_inputs(data["m"][:2], data["p"][:2], data["t"][:2], data["lnre"][:2], data["alpha"][:2],
                           data["const"][:2], variant=cfg.variant)
    net = FoilNet(K, x0.shape[-1], cfg.width, cfg.depth, cfg.variant)
    ema_net = copy.deepcopy(net)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    # 5 % warm-up (at least two steps, so that very short runs do not create an empty warm-up phase)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.steps,
                                                pct_start=max(0.05, 2.0 / cfg.steps), final_div_factor=1e3)
    t0, last = time.time(), 0.0
    loss = torch.zeros(())
    for s in range(cfg.steps):
        idx = torch.randint(0, n, (K, min(cfg.batch_size, n)))          # independent minibatch per member
        b = {k: v[idx] for k, v in data.items()}
        x, xm, base = make_inputs(b["m"], b["p"], b["t"], b["lnre"], b["alpha"], b["const"], cfg.variant)
        loss = nn.functional.smooth_l1_loss(net(x, xm, base) * LOSS_W, b["y"] * LOSS_W, beta=cfg.huber_beta)
        if cfg.residuals and col is not None:
            ci = torch.randint(0, col["m"].shape[0], (K, cfg.n_collocation_batch))
            r = physics_residuals(net, {k: v[ci] for k, v in col.items()})
            loss = loss + min(1.0, s / (0.3 * cfg.steps)) * cfg.residual_weight * (r["slope"] + r["curv"] + r["re"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():                                          # EMA of the weights over the 2nd half
            if s > cfg.steps // 2:
                for pe, pn_ in zip(ema_net.parameters(), net.parameters()):
                    pe.mul_(cfg.ema).add_(pn_, alpha=1 - cfg.ema)
            else:
                ema_net.load_state_dict(net.state_dict())
        if callback is not None:
            callback(s, float(loss.item()))
        if time.time() - last > 20 or s == cfg.steps - 1:
            last = time.time()
            el = last - t0
            log.info("[%s] step %d/%d  loss %.4f  (%.1f min, ETA %.1f min)", tag, s + 1, cfg.steps, loss.item(),
                     el / 60, el / (s + 1) * (cfg.steps - s - 1) / 60)
    return ema_net.eval()
