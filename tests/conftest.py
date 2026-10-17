"""Shared fixtures and automatic skipping of tests whose optional requirements are missing."""

from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, settings

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data"

# Deterministic property-based testing: the same examples on every run and machine.
settings.register_profile("default", derandomize=True, max_examples=60, deadline=None,
                          suppress_health_check=[HealthCheck.too_slow])
settings.register_profile("thorough", derandomize=False, max_examples=1000, deadline=None,
                          suppress_health_check=[HealthCheck.too_slow])
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))


def _have_xfoil() -> bool:
    return importlib.util.find_spec("xfoil") is not None or shutil.which("xfoil") is not None


def _have_neuralfoil() -> bool:
    return importlib.util.find_spec("neuralfoil") is not None


def _have_benchmark() -> bool:
    return (ROOT / "benchmarks" / "results" / "main_s0.json").exists() and (ROOT / "data" / "xfoil").exists()


def pytest_collection_modifyitems(config, items):
    skips = {
        "xfoil": (_have_xfoil(), "XFOIL is not installed (see docs/installation.md)"),
        "neuralfoil": (_have_neuralfoil(), "neuralfoil is not installed (pip install neuralfoil)"),
        "benchmark": (_have_benchmark(), "benchmark files not found (run from a repository checkout)"),
    }
    for item in items:
        for mark, (ok, why) in skips.items():
            if mark in item.keywords and not ok:
                item.add_marker(pytest.mark.skip(reason=why))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def pretrained():
    from neuralnaca import NeuralNACA

    return NeuralNACA.pretrained()


@pytest.fixture
def rng():
    return np.random.default_rng(1234)
