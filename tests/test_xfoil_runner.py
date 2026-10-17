"""The XFOIL driver logic, tested with stand-in solvers (no XFOIL needed)."""

import math
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from neuralnaca import xfoil as xf
from neuralnaca.xfoil import (
    XfoilConfig,
    XfoilNotFoundError,
    detect_xfoil,
    parse_polar_text,
    run_xfoil,
    sweep_chain,
)

from . import _solvers

DATA = Path(__file__).parent / "data"


def grid(n=40):
    a = np.linspace(-5, 15, n).round(3)
    return np.stack([np.full(n, 0.02), np.full(n, 0.4), np.full(n, 0.12), np.full(n, 1e6), a], 1)


def test_parse_real_polar_file():
    rows = parse_polar_text((DATA / "polar_sample.txt").read_text())
    assert rows.shape == (3, 5)
    np.testing.assert_allclose(rows[:, 0], [0.0, 2.0, 4.0])
    np.testing.assert_allclose(rows[2, [1, 2, 4]], [0.7168, 0.00702, -0.0566])


def test_parse_ignores_headers_and_garbage():
    text = " 1 1 Reynolds number fixed   Mach number fixed\n ------ ----\n   1.0  0.1  0.01  0.005  -0.02\n bad line\n"
    np.testing.assert_allclose(parse_polar_text(text), [[1.0, 0.1, 0.01, 0.005, -0.02]])
    assert parse_polar_text("").shape == (0, 5)


@pytest.mark.parametrize(("alpha", "chain"), [
    (12.3, [2.3, 7.3, 12.3]), (5.0, [5.0]), (4.0, [4.0]), (5.001, [0.001, 5.001]), (0.0, [0.0]),
    (-5.0, [0.0, -5.0]), (-7.0, [-2.0, -7.0]), (-0.5, [-0.5]), (15.0, [5.0, 10.0, 15.0]),
])
def test_sweep_chain_examples(alpha, chain):
    assert sweep_chain(alpha) == pytest.approx(chain)


@given(st.floats(-20, 20, allow_nan=False), st.sampled_from([1.0, 2.5, 5.0]))
def test_sweep_chain_properties(alpha, step):
    """Mirrors asb.XFoil.alpha(start_at=0): same side of zero, start next to zero, fixed spacing, target last."""
    c = sweep_chain(alpha, step)
    assert c[-1] == pytest.approx(round(alpha, 6))
    assert np.allclose(np.abs(np.diff(c)), step)
    if alpha > 0:                      # (angles are rounded to 1e-6 deg, so a tiny positive angle may become 0.0)
        assert all(a >= 0 for a in c) and c[0] <= step + 1e-9
    else:
        assert all(a <= 0 for a in c) and c[0] > -step - 1e-9


def test_config_validation_and_tags():
    assert XfoilConfig().tag() == "p279_nc9"
    nf = XfoilConfig().neuralfoil_protocol(geometry="kulfan")
    assert (nf.protocol, nf.max_iter, nf.geometry) == ("nf-sweep", 200, "kulfan")
    assert nf.tag() == "p279_nc9_nf-sweep_kulfan_it200"
    for bad in ({"backend": "x"}, {"protocol": "warm"}, {"geometry": "cst"}, {"backend": "binary"}):
        with pytest.raises(ValueError):
            XfoilConfig(**bad)


def test_run_matches_direct_evaluation_in_parallel_and_in_process():
    pts = grid()
    expected = np.array([_solvers.analytic_teacher(p, None) for p in pts])
    np.testing.assert_allclose(run_xfoil(pts, XfoilConfig(), 2, chunk=7, solver=_solvers.analytic_teacher), expected)
    np.testing.assert_allclose(run_xfoil(pts, XfoilConfig(), 0, solver=_solvers.analytic_teacher), expected)


def test_worker_crashes_are_contained():
    """Points that kill their worker process become NaN; every other point is still solved correctly."""
    pts = grid(60)
    pts[17, 4], pts[45, 4] = 13.0, 7.0
    out = run_xfoil(pts, XfoilConfig(), 2, chunk=16, solver=_solvers.crashing_teacher)
    bad = ~np.isfinite(out[:, 0])
    assert set(np.where(bad)[0]) == {17, 45}
    expected = np.array([_solvers.analytic_teacher(p, None) for p in pts])
    np.testing.assert_allclose(out[~bad], expected[~bad])


def test_exceptions_and_unconverged_points_become_nan():
    pts = grid()
    out = run_xfoil(pts, XfoilConfig(), 0, solver=_solvers.raising_teacher)
    assert np.isnan(out[pts[:, 4] < 0]).all() and np.isfinite(out[pts[:, 4] >= 0]).all()
    out = run_xfoil(pts, XfoilConfig(), 0, solver=_solvers.nonconverging_teacher)
    assert np.isnan(out[pts[:, 4] > 10]).all() and np.isfinite(out[pts[:, 4] <= 10]).all()


def test_solve_point_maps_backend_failures_to_nan(monkeypatch):
    def broken(pt, cfg):
        raise OSError("xfoil exploded")

    monkeypatch.setattr(xf, "_module_solver", broken)
    assert all(math.isnan(v) for v in xf.solve_point((0.0, 0.0, 0.12, 1e6, 0.0), XfoilConfig()))
    monkeypatch.setattr(xf, "_module_solver", lambda pt, cfg: (0.3, -0.01, 0.0))
    assert all(math.isnan(v) for v in xf.solve_point((0.0, 0.0, 0.12, 1e6, 0.0), XfoilConfig()))


def test_detect_reports_missing_xfoil_with_instructions(monkeypatch):
    import builtins
    import shutil

    real_import = builtins.__import__

    def no_xfoil(name, *a, **k):
        if name == "xfoil":
            raise ImportError
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_xfoil)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.delenv("NEURALNACA_XFOIL", raising=False)
    with pytest.raises(XfoilNotFoundError, match="install_xfoil_python"):
        detect_xfoil()
    with pytest.raises(XfoilNotFoundError):
        detect_xfoil("module")
    cfg = detect_xfoil("binary", executable="/opt/xfoil")
    assert cfg.backend == "binary" and cfg.executable == "/opt/xfoil"
    monkeypatch.setenv("NEURALNACA_XFOIL", "/env/xfoil")
    assert detect_xfoil().executable == "/env/xfoil"
    with pytest.raises(ValueError):
        detect_xfoil("gpu")


def test_binary_solver_runs_the_expected_keystrokes(monkeypatch, tmp_path):
    """Without XFOIL: a fake executable checks the command script and writes a polar file."""
    captured = {}

    def fake_run(cmd, input, **kw):
        captured["input"] = input
        lines = input.splitlines()
        pol = lines[lines.index("PACC") + 1]
        rows = "".join(f"  {float(l.split()[1]):7.3f}   0.5000   0.01000   0.00500  -0.0500\n"
                       for l in lines if l.startswith("ALFA"))
        Path(pol).write_text("header\n" + rows)

    monkeypatch.setattr(xf.subprocess, "run", fake_run)
    cfg = XfoilConfig(backend="binary", executable="xfoil", protocol="nf-sweep")
    out = xf._binary_solver((0.02, 0.4, 0.12, 1e6, 12.3), cfg)
    assert out == pytest.approx((0.5, 0.01, -0.05))
    script = captured["input"]
    assert "N 279" in script and "VISC 1.000000e+06" in script and "N 9.0" in script and "ITER 100" in script
    assert [l for l in script.splitlines() if l.startswith("ALFA")] == ["ALFA 2.3000", "ALFA 7.3000", "ALFA 12.3000"]


def test_binary_solver_reports_unconverged_target(monkeypatch):
    def fake_run(cmd, input, **kw):
        lines = input.splitlines()
        Path(lines[lines.index("PACC") + 1]).write_text("   2.300   0.5  0.01  0.005  -0.05\n")   # target missing

    monkeypatch.setattr(xf.subprocess, "run", fake_run)
    cfg = XfoilConfig(backend="binary", executable="xfoil", protocol="nf-sweep")
    assert all(math.isnan(v) for v in xf._binary_solver((0.02, 0.4, 0.12, 1e6, 12.3), cfg))


@pytest.mark.neuralfoil
def test_kulfan_equivalent_is_close_to_the_naca_section():
    co, frame = xf.kulfan_equivalent(0.04, 0.4, 0.12)
    assert co.shape[1] == 2 and len(co) > 300
    assert abs(frame["delta_alpha"]) < 0.5 and abs(frame["scale"] - 1) < 0.01
    from neuralnaca.geometry import naca4_coordinates

    ref = naca4_coordinates(0.04, 0.4, 0.12, 400)
    up = co[: len(co) // 2]
    assert np.interp(0.3, up[::-1, 0], up[::-1, 1]) == pytest.approx(np.interp(0.3, ref[:400][::-1, 0],
                                                                                ref[:400][::-1, 1]), abs=0.003)


@pytest.mark.neuralfoil
def test_normalized_equivalent_keeps_the_exact_shape():
    _, frame = xf.normalized_equivalent(0.0, 0.0, 0.12)
    assert abs(frame["delta_alpha"]) < 1e-9 and frame["scale"] == pytest.approx(1.0)
    _, frame2 = xf.normalized_equivalent(0.04, 0.4, 0.12)
    assert 0.05 < frame2["delta_alpha"] < 0.5                  # cambered NACA sections are rotated slightly
    assert XfoilConfig(geometry="naca-normalized").tag() == "p279_nc9_cold_naca-normalized_it100"
