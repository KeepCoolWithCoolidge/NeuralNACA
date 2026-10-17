import dataclasses

import numpy as np
import pytest
import torch

from neuralnaca.domain import sample_points
from neuralnaca.model import predict_tensors, to_tensors
from neuralnaca.panel import airfoil_constants
from neuralnaca.training import TrainConfig, physics_residuals, train_ensemble

from ._solvers import analytic_teacher


def synthetic(n, kind="sobol", seed=0):
    pts = sample_points(n, kind, seed)
    res = np.array([analytic_teacher(p, None) for p in pts])
    return pts, res


@pytest.fixture(scope="module")
def data():
    pts, res = synthetic(256)
    return to_tensors(pts, airfoil_constants(pts[:, :3]), res)


def test_documented_defaults_are_the_reported_configuration():
    """Guard against silent changes of the hyper-parameters behind every reported number."""
    assert TrainConfig().to_dict() == {
        "ensemble": 5, "steps": 18000, "batch_size": 512, "lr": 3e-3, "huber_beta": 0.1, "ema": 0.9995,
        "width": 128, "depth": 4, "seed": 0, "variant": "full", "residuals": False, "residual_weight": 1.0,
        "n_collocation_batch": 128}


def test_config_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        TrainConfig().steps = 3  # type: ignore[misc]


def test_training_reduces_the_loss_and_fits(data):
    losses = []
    cfg = TrainConfig(ensemble=2, steps=400, width=32, depth=2)
    net = train_ensemble(data, cfg, callback=lambda s, l: losses.append(l))
    assert len(losses) == 400
    assert np.mean(losses[-40:]) < 0.5 * np.mean(losses[:40])
    pts, res = synthetic(128, "random", 5)
    pred = predict_tensors(net, to_tensors(pts, airfoil_constants(pts[:, :3]))).mean(0)
    assert np.abs(pred[:, 0] - res[:, 0]).mean() < 0.1


def test_training_is_deterministic_for_a_seed(data):
    cfg = TrainConfig(ensemble=2, steps=30, width=16, depth=2, seed=7)
    a, b = train_ensemble(data, cfg), train_ensemble(data, cfg)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
    c = train_ensemble(data, dataclasses.replace(cfg, seed=8))
    assert any(not torch.equal(pa, pc) for pa, pc in zip(a.parameters(), c.parameters()))


@pytest.mark.parametrize("variant", ["plain", "no_panel", "no_symmetry", "no_drag_prior"])
def test_every_variant_trains(data, variant):
    net = train_ensemble(data, TrainConfig(ensemble=1, steps=20, width=16, depth=2, variant=variant))
    assert net.variant == variant


def test_physics_residuals_are_finite_and_trainable(data):
    pts = sample_points(64, "sobol", 3)
    col = to_tensors(pts, airfoil_constants(pts[:, :3]))
    net = train_ensemble(data, TrainConfig(ensemble=2, steps=15, width=16, depth=2, residuals=True), col=col)
    r = physics_residuals(net, {k: v[None].expand(2, *v.shape) for k, v in col.items()})
    assert set(r) == {"slope", "curv", "re"}
    assert all(torch.isfinite(v) and v >= 0 for v in r.values())


def test_residuals_need_collocation_points(data):
    with pytest.raises(ValueError):
        train_ensemble(data, TrainConfig(steps=1, residuals=True))
