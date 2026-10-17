"""Validation of the differentiable panel method against exact potential-flow solutions and XFOIL."""

import json

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from neuralnaca.geometry import naca4_nodes, parse_naca
from neuralnaca.panel import airfoil_constants, constants_from_nodes, hess_smith, inviscid, inviscid_numpy


def joukowski_nodes(mu_x: float, mu_y: float, n: int):
    """Clockwise nodes of a Joukowski airfoil (circle through z = 1 mapped by z + 1/z), normalised to unit chord.

    Returns the nodes, the circle radius / chord and the zero-lift angle beta; the exact lift coefficient is
    ``CL = 8 pi (R / c) sin(alpha + beta)``.
    """
    zc = complex(mu_x, mu_y)
    R = abs(1 - zc)
    th0 = np.angle(1 - zc)
    zeta = zc + R * np.exp(1j * (th0 - np.linspace(0, 2 * np.pi, n + 1)))
    z = zeta + 1 / zeta
    x0, c = z.real.min(), z.real.max() - z.real.min()
    nodes = np.stack([(z.real - x0) / c, z.imag / c], 1)
    return torch.tensor(nodes)[None], R / c, -th0


@pytest.mark.parametrize(("mu", "alpha"), [((-0.1, 0.0), 5.0), ((-0.08, 0.08), 0.0), ((-0.08, 0.08), 5.0),
                                           ((-0.05, 0.1), 3.0)])
def test_joukowski_exact_lift_converges(mu, alpha):
    """Constant-source panels converge (slowly, because of the cusped trailing edge) to the exact solution."""
    errs = []
    for n in (120, 240, 480):
        nodes, R_c, beta = joukowski_nodes(*mu, n)
        C = constants_from_nodes(nodes).numpy()[0]
        a = np.radians(alpha)
        exact = 8 * np.pi * R_c * np.sin(a + beta)
        errs.append(abs(C[0] * np.cos(a) + C[1] * np.sin(a) - exact) / abs(exact))
    assert errs[0] < 0.04
    assert errs[2] < errs[1] < errs[0]                       # monotone convergence
    assert np.log2(errs[0] / errs[2]) / 2 > 0.5              # observed order > 0.5 (theory for a cusp: ~0.7)


def test_symmetric_joukowski_has_no_lift_at_zero_incidence():
    nodes, _, _ = joukowski_nodes(-0.1, 0.0, 200)
    C = constants_from_nodes(nodes).numpy()[0]
    assert abs(C[0]) < 1e-12 and abs(C[2]) < 1e-12


REF = json.loads(
    (__import__("pathlib").Path(__file__).parent / "data" / "xfoil_inviscid.json").read_text(encoding="utf-8")
)


@pytest.mark.parametrize("code", sorted(REF["sections"]))
def test_matches_xfoil_inviscid(code):
    """Against XFOIL 6.99's own inviscid solution (279 panels) for the same coordinates (fixture in tests/data)."""
    r = REF["sections"][code]["279"]
    c = airfoil_constants(np.array([parse_naca(code)]))
    cl, cm = inviscid_numpy(np.repeat(c, len(r["alpha"]), 0), np.array(r["alpha"]))
    np.testing.assert_array_less(np.abs(cl - r["CL"]), 0.02 * np.maximum(np.abs(r["CL"]), 1.0))
    np.testing.assert_array_less(np.abs(cm - r["CM"]), 0.0065)


def test_lift_from_circulation_equals_lift_from_pressure():
    """Kutta-Joukowski lift and integrated surface-pressure lift are two independent outputs of the solver."""
    m, p, t = (torch.tensor([v], dtype=torch.float64) for v in (0.04, 0.4, 0.12))
    s = hess_smith(naca4_nodes(m, p, t, 100))
    for k, alpha in ((0, 0.0), (1, 90.0)):          # basis solutions: free stream along x, along y
        cp = 1 - s["Vt"][0, :, k] ** 2
        force = -(cp[:, None] * s["n"][0] * s["S"][0, :, None]).sum(0)
        a = np.radians(alpha)
        lift = float(-force[0] * np.sin(a) + force[1] * np.cos(a))
        assert lift == pytest.approx(float(-2 * s["circ"][0, k]), rel=0.02, abs=0.01)


def test_thin_airfoil_theory_limits():
    """Lift slope ~ 2 pi (1 + 0.77 t) per rad; NACA 2412 zero-lift angle close to thin-airfoil theory (-2.08 deg)."""
    c = airfoil_constants(np.array([[0.0, 0.0, 0.12], [0.02, 0.4, 0.12]]))
    slope = c[0, 1]                                             # dCL/dalpha at alpha = 0 for the symmetric section
    assert slope == pytest.approx(2 * np.pi * (1 + 0.77 * 0.12), rel=0.03)
    a0 = np.degrees(np.arctan2(-c[1, 0], c[1, 1]))
    assert a0 == pytest.approx(-2.08, abs=0.25)


@given(st.floats(0.0, 0.08), st.floats(0.2, 0.7), st.floats(0.06, 0.18))
def test_reflection_identities_of_the_constants(m, p, t):
    """Reflecting the section maps CL(a) -> -CL(-a), CM(a) -> -CM(-a); hence these sign rules, which hold up to the
    round-off of the dense solve (~1e-7 on constants of order 0.1-7)."""
    c, r = airfoil_constants(np.array([[m, p, t], [-m, p, t]]))
    np.testing.assert_allclose(r, c * np.array([-1, 1, -1, -1, 1, -1]), atol=2e-6)


def closed_te_nodes(m, p, t, n_half):
    """NACA section with the closed-trailing-edge thickness coefficient (-0.1036 instead of -0.1015)."""
    x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n_half + 1)))
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)
    yc = np.where(x < p, m / p ** 2 * (2 * p * x - x ** 2), m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))
    th = np.arctan(np.where(x < p, 2 * m / p ** 2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x)))
    xu, yu, xl, yl = x - yt * np.sin(th), yc + yt * np.cos(th), x + yt * np.sin(th), yc - yt * np.cos(th)
    return torch.tensor(np.stack([np.concatenate([xl[::-1], xu[1:]]), np.concatenate([yl[::-1], yu[1:]])], 1))[None]


def test_converges_under_refinement_for_closed_trailing_edge():
    """Solver consistency: for a closed body the lift converges (differences shrink ~2x per panel doubling)."""
    a = np.radians(4.0)
    cl = []
    for n in (60, 120, 240, 480):
        C = constants_from_nodes(closed_te_nodes(0.04, 0.4, 0.12, n)).numpy()[0]
        cl.append(C[0] * np.cos(a) + C[1] * np.sin(a))
    d = np.abs(np.diff(cl))
    assert d[2] < d[1] < d[0] and d[2] < 1e-3
    assert cl[-1] == pytest.approx(1.002, abs=0.003)          # XFOIL 6.99 inviscid, NACA 4412, 4 deg


def test_open_trailing_edge_drift_is_bounded_at_model_resolution():
    """Known limitation (documented): with the open NACA trailing edge the discrete Kutta condition is applied across
    the gap, and the solution drifts slowly (~ -0.7 % per panel doubling) instead of converging. The model uses a
    *fixed* 120-panel discretisation, where the inviscid lift is within 2 % of XFOIL's
    (see test_matches_xfoil_inviscid);
    this test pins the size of the drift so that a change in behaviour is noticed."""
    c60 = airfoil_constants(np.array([[0.04, 0.4, 0.12]]), n_half=60)
    c240 = airfoil_constants(np.array([[0.04, 0.4, 0.12]]), n_half=240)
    cl60, _ = inviscid_numpy(c60, np.array([4.0]))
    cl240, _ = inviscid_numpy(c240, np.array([4.0]))
    assert 0.005 < (cl60 - cl240)[0] / cl240[0] < 0.02


def test_inviscid_is_differentiable_in_alpha():
    c = torch.tensor(airfoil_constants(np.array([[0.02, 0.4, 0.12]])))
    a = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: torch.stack(inviscid(c, x)), (a,))


def test_batching_is_consistent():
    pts = np.array([[0.02, 0.4, 0.12], [0.0, 0.5, 0.09], [0.06, 0.3, 0.15]])
    np.testing.assert_allclose(airfoil_constants(pts, chunk=1), airfoil_constants(pts, chunk=256), rtol=1e-12)
    assert airfoil_constants(np.zeros((0, 3))).shape == (0, 6)


def test_camber_position_irrelevant_without_camber():
    a = airfoil_constants(np.array([[0.0, 0.2, 0.12], [0.0, 0.7, 0.12]]))
    np.testing.assert_allclose(a[0], a[1], atol=1e-14)
