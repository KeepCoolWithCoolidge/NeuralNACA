r"""Differentiable Hess-Smith panel method (the "physics" backbone of the surrogate).

Inviscid, incompressible potential flow around the section is represented by a constant-strength source on every
panel plus one vortex strength shared by all panels; the flow-tangency condition at the panel midpoints and the
Kutta condition (equal tangential velocity magnitude on the two trailing-edge panels) close the linear system.

Because potential flow is linear in the free-stream vector, two unit solutions (free stream along *x* and along
*y*) are enough to express the inviscid lift and moment analytically for *any* angle of attack:

.. math::

    C_{L,\mathrm{inv}}(\alpha) = A_L\cos\alpha + B_L\sin\alpha, \qquad
    C_{M,\mathrm{inv}}(\alpha) = A_M - (B_M\cos^2\alpha + C_M\cos\alpha\sin\alpha + D_M\sin^2\alpha)

with :math:`C_L` from the Kutta-Joukowski theorem (:math:`C_L = 2\Gamma/(V_\infty c)`) and :math:`C_M` about the
quarter chord from integration of :math:`C_p = 1 - V_t^2/V_\infty^2`. The six constants depend only on geometry and
are computed once per airfoil by :func:`airfoil_constants`; :func:`inviscid` then evaluates them (differentiably in
``alpha``) inside the network.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .geometry import naca4_nodes

__all__ = ["DEG", "airfoil_constants", "hess_smith", "inviscid", "inviscid_numpy"]

DEG = math.pi / 180.0


def hess_smith(nodes: torch.Tensor) -> dict[str, torch.Tensor]:
    """Solve the Hess-Smith system for a batch of closed or open polygons.

    Parameters
    ----------
    nodes
        ``(B, N + 1, 2)`` panel end points in *clockwise* order (lower trailing edge -> leading edge -> upper
        trailing edge), chord normalised to 1.

    Returns
    -------
    dict
        ``S`` panel lengths ``(B, N)``, ``n`` exterior unit normals ``(B, N, 2)``, ``mid`` midpoints ``(B, N, 2)``,
        ``Vt`` tangential surface velocity at the midpoints for the two unit free streams ``(B, N, 2)``,
        ``circ`` total circulation for the two unit free streams ``(B, 2)``.
    """
    dtype = nodes.dtype
    P1, P2 = nodes[:, :-1], nodes[:, 1:]
    d = P2 - P1
    S = torch.linalg.norm(d, dim=-1)
    t_vec = d / S[..., None]
    n_vec = torch.stack([-t_vec[..., 1], t_vec[..., 0]], -1)           # exterior normal (clockwise ordering)
    mid = 0.5 * (P1 + P2)
    B, N = S.shape
    rel = mid[:, :, None, :] - P1[:, None, :, :]
    xl = (rel * t_vec[:, None, :, :]).sum(-1)
    zl = (rel * n_vec[:, None, :, :]).sum(-1)
    Sj = S[:, None, :]
    eye = torch.eye(N, dtype=torch.bool)[None]
    r1sq = torch.where(eye, torch.ones_like(xl), xl ** 2 + zl ** 2)
    r2sq = torch.where(eye, torch.ones_like(xl), (xl - Sj) ** 2 + zl ** 2)
    lnr = torch.where(eye, torch.zeros_like(xl), 0.5 * torch.log(r1sq / r2sq))
    dth = torch.where(eye, torch.full_like(xl, math.pi), torch.atan2(zl, xl - Sj) - torch.atan2(zl, xl))
    two_pi = 2.0 * math.pi
    us, ws = lnr / two_pi, dth / two_pi                                # unit source, panel-local frame
    uv, wv = -dth / two_pi, lnr / two_pi                               # unit vortex, panel-local frame

    def glob(u, w):
        return u[..., None] * t_vec[:, None, :, :] + w[..., None] * n_vec[:, None, :, :]

    Vs, Vv = glob(us, ws), glob(uv, wv)
    An = (Vs * n_vec[:, :, None, :]).sum(-1)
    At = (Vs * t_vec[:, :, None, :]).sum(-1)
    Bn = (Vv * n_vec[:, :, None, :]).sum(-1).sum(-1)
    Bt = (Vv * t_vec[:, :, None, :]).sum(-1).sum(-1)
    M = torch.zeros(B, N + 1, N + 1, dtype=dtype)
    M[:, :N, :N], M[:, :N, N] = An, Bn
    M[:, N, :N] = At[:, 0, :] + At[:, N - 1, :]                        # Kutta: Vt(first) + Vt(last) = 0
    M[:, N, N] = Bt[:, 0] + Bt[:, N - 1]
    rhs = torch.zeros(B, N + 1, 2, dtype=dtype)
    for k, vinf in enumerate(([1.0, 0.0], [0.0, 1.0])):
        v = torch.tensor(vinf, dtype=dtype)
        rhs[:, :N, k] = -(n_vec * v).sum(-1)
        rhs[:, N, k] = -((t_vec[:, 0] * v).sum(-1) + (t_vec[:, N - 1] * v).sum(-1))
    sol = torch.linalg.solve(M, rhs)
    sigma, gamma = sol[:, :N, :], sol[:, N, :]
    Vt = torch.stack([(t_vec * torch.tensor(v, dtype=dtype)).sum(-1) + (At * sigma[:, :, k][:, None, :]).sum(-1)
                      + Bt * gamma[:, k][:, None] for k, v in enumerate(([1.0, 0.0], [0.0, 1.0]))], -1)
    return {"S": S, "n": n_vec, "mid": mid, "Vt": Vt, "circ": gamma * S.sum(-1, keepdim=True)}


def constants_from_nodes(nodes: torch.Tensor) -> torch.Tensor:
    """The six inviscid constants ``[A_L, B_L, A_M, B_M, C_M, D_M]`` for arbitrary clockwise node sets.

    See the module docstring for their meaning. Moments are taken about ``(x, y) = (0.25, 0)``.
    """
    s = hess_smith(nodes)
    circ, V, S, n, mid = s["circ"], s["Vt"], s["S"], s["n"], s["mid"]
    w = ((mid[..., 0] - 0.25) * n[..., 1] - mid[..., 1] * n[..., 0]) * S
    return torch.stack([-2 * circ[:, 0], -2 * circ[:, 1], w.sum(-1), (w * V[..., 0] ** 2).sum(-1),
                        2 * (w * V[..., 0] * V[..., 1]).sum(-1), (w * V[..., 1] ** 2).sum(-1)], 1)


def airfoil_constants(mpt: np.ndarray, chunk: int = 256, n_half: int = 60) -> np.ndarray:
    """Inviscid constants for a batch of NACA 4-digit sections.

    Parameters
    ----------
    mpt
        ``(n, 3)`` array of ``[m, p, t]``. Negative ``m`` (reflected sections) is allowed.
    chunk
        Batch size of the dense panel solves (memory ~ ``chunk * (2 n_half)^2``).
    n_half
        Panels per surface (60 -> 120 panels; see the convergence test in the test suite).

    Returns
    -------
    numpy.ndarray, shape (n, 6), float64
        ``[A_L, B_L, A_M, B_M, C_M, D_M]`` such that, with ``alpha`` in radians,
        ``CL_inv = A_L cos(alpha) + B_L sin(alpha)`` and
        ``CM_inv = A_M - (B_M cos^2 + C_M cos sin + D_M sin^2)``.
    """
    mpt = np.asarray(mpt, dtype=np.float64).reshape(-1, 3)
    out = []
    with torch.no_grad():
        for i in range(0, len(mpt), chunk):
            m, p, t = (torch.tensor(mpt[i:i + chunk, j], dtype=torch.float64) for j in range(3))
            out.append(constants_from_nodes(naca4_nodes(m, p, t, n_half)).numpy())
    return np.concatenate(out) if out else np.zeros((0, 6))


def inviscid(const: torch.Tensor, alpha_deg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate inviscid ``(CL, CM)`` from the constants (differentiable in ``alpha_deg``, in degrees)."""
    a = alpha_deg * DEG
    c, s = torch.cos(a), torch.sin(a)
    cl = const[..., 0] * c + const[..., 1] * s
    cm = const[..., 2] - (const[..., 3] * c * c + const[..., 4] * c * s + const[..., 5] * s * s)
    return cl, cm


def inviscid_numpy(const: np.ndarray, alpha_deg) -> tuple[np.ndarray, np.ndarray]:
    """NumPy version of :func:`inviscid`."""
    cl, cm = inviscid(torch.as_tensor(np.asarray(const, dtype=np.float64)),
                      torch.as_tensor(np.asarray(alpha_deg, dtype=np.float64)))
    return cl.numpy(), cm.numpy()
