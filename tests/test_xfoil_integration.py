"""Tests that run the real XFOIL (skipped when XFOIL is not installed)."""

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from neuralnaca.xfoil import XfoilConfig, detect_xfoil, run_xfoil

pytestmark = pytest.mark.xfoil

REF = json.loads((Path(__file__).parent / "data" / "xfoil_viscous_reference.json").read_text(encoding="utf-8"))


def backends():
    out = []
    try:
        import xfoil  # noqa: F401

        out.append(XfoilConfig())
    except ImportError:
        pass
    exe = os.environ.get("NEURALNACA_XFOIL") or shutil.which("xfoil")
    if exe:
        out.append(XfoilConfig(backend="binary", executable=exe))
    return out


@pytest.fixture(params=backends() or [None], ids=lambda c: c.backend if c else "none")
def cfg(request):
    if request.param is None:
        pytest.skip("no XFOIL backend")
    return request.param


def test_reproduces_the_shipped_xfoil_data(cfg):
    """Your XFOIL agrees with the build that produced the shipped data on 24 attached-flow points."""
    pts, ref = np.array(REF["points"]), np.array(REF["results"])
    got = run_xfoil(pts, cfg, 2)
    ok = np.isfinite(got[:, 0])
    assert ok.mean() >= 0.95
    d = np.abs(got[ok] - ref[ok])
    assert np.median(d[:, 0]) < 1e-3 and (d[:, 0] < 0.01).mean() >= 0.9
    assert np.median(np.abs(np.log(got[ok, 1] / ref[ok, 1]))) < 0.01


def test_sweep_protocol_equals_cold_start_for_single_step_chains(cfg):
    pts = np.array([[0.02, 0.4, 0.12, 1e6, 3.0], [0.0, 0.0, 0.10, 5e5, -2.0]])          # chains of length 1
    a = run_xfoil(pts, cfg, 1)
    b = run_xfoil(pts, replace(cfg, protocol="nf-sweep"), 1)
    np.testing.assert_allclose(a, b, atol=2e-4)


def test_sweep_protocol_converges_for_long_chains(cfg):
    pts = np.array([[0.02, 0.4, 0.12, 1e6, 12.0], [0.04, 0.4, 0.15, 3e6, 14.0]])
    out = run_xfoil(pts, cfg.neuralfoil_protocol(), 1)
    assert np.isfinite(out).all() and (out[:, 0] > 1.0).all()


@pytest.mark.neuralfoil
def test_kulfan_geometry_is_close_to_exact_geometry(cfg):
    pts = np.array([[0.02, 0.4, 0.12, 1e6, 2.0], [0.0, 0.0, 0.12, 2e6, 4.0]])
    a = run_xfoil(pts, cfg, 1)
    b = run_xfoil(pts, replace(cfg, geometry="kulfan"), 1)
    np.testing.assert_allclose(a[:, 0], b[:, 0], atol=0.02)
    np.testing.assert_allclose(np.log(a[:, 1]), np.log(b[:, 1]), atol=0.05)


def test_detect_finds_a_backend():
    assert detect_xfoil().backend in ("module", "binary")


@pytest.mark.neuralfoil
def test_neuralfoil_frame_transformation_alone_barely_changes_xfoil(cfg):
    """Control for the ground-truth study: the exact shape in NeuralFoil's normalised frame gives (almost) the same
    XFOIL answer, so differences between `naca` and `kulfan` truths come from the Kulfan shape fit."""
    pts = np.array([[0.04, 0.4, 0.12, 1e6, 2.0], [0.02, 0.3, 0.10, 3e5, 5.0], [0.06, 0.5, 0.15, 3e6, -2.0]])
    a = run_xfoil(pts, cfg, 1)
    b = run_xfoil(pts, replace(cfg, geometry="naca-normalized"), 1)
    np.testing.assert_allclose(a[:, 0], b[:, 0], atol=0.004)
    np.testing.assert_allclose(a[:, 2], b[:, 2], atol=0.003)
