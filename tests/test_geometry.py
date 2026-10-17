import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from neuralnaca.geometry import format_naca, naca4_coordinates, naca4_nodes, parse_naca


@pytest.mark.parametrize(("code", "expected"), [
    ("2412", (0.02, 0.4, 0.12)), ("NACA 2412", (0.02, 0.4, 0.12)), ("naca0012", (0.0, 0.0, 0.12)),
    ("NACA-4415", (0.04, 0.4, 0.15)), ("  6409 ", (0.06, 0.4, 0.09)), ("0412", (0.0, 0.0, 0.12)),
])
def test_parse_naca_accepts_common_spellings(code, expected):
    assert parse_naca(code) == pytest.approx(expected)


@pytest.mark.parametrize("code", ["241", "24121", "NACA 23012", "2a12", "", "2012", "2400", "NACA"])
def test_parse_naca_rejects_invalid(code):
    with pytest.raises(ValueError):
        parse_naca(code)


@given(st.integers(0, 9), st.integers(1, 9), st.integers(1, 99))
def test_format_parse_round_trip(dm, dp, dt):
    code = f"{dm}{dp if dm else 0}{dt:02d}"
    assert format_naca(*parse_naca(code)) == code


def test_format_naca_rejects_non_integer_sections():
    with pytest.raises(ValueError):
        format_naca(0.025, 0.4, 0.12)


def test_coordinates_basic_shape_and_ordering():
    co = naca4_coordinates(0.02, 0.4, 0.12, n_per_side=100)
    assert co.shape == (199, 2)
    # TE upper ... TE lower (thickness is applied normal to the camber line, so x deviates from 1 by ~1e-4)
    assert co[0, 0] == pytest.approx(1.0, abs=1e-3) and co[-1, 0] == pytest.approx(1.0, abs=1e-3)
    assert co[99] == pytest.approx([0.0, 0.0], abs=1e-12)                           # leading edge
    assert co[:99, 1].mean() > co[100:, 1].mean()                                   # upper surface first (Selig)


def test_symmetric_section_thickness_and_symmetry():
    co = naca4_coordinates(0.0, 0.0, 0.12, n_per_side=201)
    up, lo = co[:201][::-1], co[200:]
    np.testing.assert_allclose(up[:, 1], -lo[:, 1], atol=1e-15)
    thick = 2 * up[:, 1]
    assert thick.max() == pytest.approx(0.12, rel=2e-3)                            # NACA definition: t = max thickness
    assert up[np.argmax(thick), 0] == pytest.approx(0.30, abs=0.01)                 # at 30 % chord
    assert 2 * up[-1, 1] == pytest.approx(0.12 * 0.0021 / 0.20 * 2, rel=1e-6)      # open trailing edge: 0.252 % t


def test_camber_line_maximum():
    co = naca4_coordinates(0.04, 0.4, 0.12, n_per_side=400)
    up, lo = co[:400][::-1], co[399:]
    # thickness is applied normal to the camber line, so the surface midpoint approximates the camber line
    mid = 0.5 * (up[:, 1] + lo[:, 1])
    assert mid.max() == pytest.approx(0.04, rel=0.01)


@given(st.floats(0.005, 0.08), st.floats(0.2, 0.7), st.floats(0.06, 0.18))
def test_negative_camber_is_mirror_image(m, p, t):
    a, b = naca4_coordinates(m, p, t), naca4_coordinates(-m, p, t)
    # reflected section: same x stations; upper surface of b == mirrored lower surface of a
    n = (len(a) + 1) // 2
    np.testing.assert_allclose(b[:n][::-1, 1], -a[n - 1:, 1], atol=1e-14)


@given(st.floats(0.0, 0.08), st.floats(0.2, 0.7), st.floats(0.06, 0.18))
def test_panel_nodes_match_coordinates(m, p, t):
    """The torch nodes (clockwise) are the reversed numpy coordinates for the same spacing.

    The only difference is at the leading edge, where the torch version evaluates sqrt(max(x, 1e-12)) to keep the
    gradient finite (a 1.9e-7 offset of the leading-edge node)."""
    nodes = naca4_nodes(torch.tensor([m], dtype=torch.float64), torch.tensor([p], dtype=torch.float64),
                        torch.tensor([t], dtype=torch.float64), n_half=40)[0].numpy()
    co = naca4_coordinates(m, p, t, n_per_side=41)
    np.testing.assert_allclose(nodes, co[::-1], atol=3e-7)


def test_nodes_are_differentiable():
    m = torch.tensor([0.03], dtype=torch.float64, requires_grad=True)
    nodes = naca4_nodes(m, torch.tensor([0.4], dtype=torch.float64), torch.tensor([0.12], dtype=torch.float64))
    nodes[..., 1].sum().backward()
    assert m.grad is not None and torch.isfinite(m.grad).all()


def test_cambered_section_needs_valid_p():
    with pytest.raises(ValueError):
        naca4_coordinates(0.02, 0.0, 0.12)
