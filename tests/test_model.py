import json

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

import neuralnaca.model as model_mod
from neuralnaca.model import (
    FEATURES_VERSION,
    VARIANTS,
    FoilNet,
    IncompatibleWeightsError,
    check_reflection,
    drag_prior,
    load_model,
    make_inputs,
    n_features,
    predict_tensors,
    save_model,
    to_tensors,
)
from neuralnaca.panel import airfoil_constants


def random_net(variant="full", K=3, seed=0, scale=1.0):
    torch.manual_seed(seed)
    net = FoilNet(K, n_features(variant), 32, 3, variant).eval()
    with torch.no_grad():
        for W in net.f.W:
            W.mul_(scale)
    return net


def packed(points):
    points = np.atleast_2d(np.asarray(points, float))
    return to_tensors(points, airfoil_constants(points[:, :3]))


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_feature_shapes(variant):
    d = packed([[0.02, 0.4, 0.12, 1e6, 3.0], [0.0, 0.3, 0.09, 2e5, -2.0]])
    x, xm, base = make_inputs(d["m"], d["p"], d["t"], d["lnre"], d["alpha"], d["const"], variant)
    assert x.shape == (2, n_features(variant)) and base.shape == (2, 3)
    assert (xm is not None) == VARIANTS[variant].symmetric


def test_unknown_variant_and_wrong_width_are_rejected():
    with pytest.raises(ValueError):
        n_features("bogus")
    with pytest.raises(ValueError):
        FoilNet(2, 5, variant="full")


@given(st.floats(-0.08, 0.08), st.floats(0.2, 0.7))
def test_camber_position_gate_is_even_in_camber(m, p):
    """The bug fixed in feature version 2: the gate must not depend on the sign of the camber."""
    t = torch.tensor
    one = t([1.0])
    x1, _, _ = make_inputs(t([m]), t([p]), 0.12 * one, 13.8 * one, one, torch.zeros(1, 6))
    x2, _, _ = make_inputs(t([-m]), t([p]), 0.12 * one, 13.8 * one, one, torch.zeros(1, 6))
    assert torch.equal(x1[:, 1], x2[:, 1])


@given(st.floats(0.0, 0.08), st.floats(0.2, 0.7), st.floats(0.06, 0.18), st.floats(5.0, 7.0), st.floats(-5.0, 15.0),
       st.integers(0, 3))
def test_exact_reflection_symmetry_for_random_networks(m, p, t, lre, alpha, seed):
    """Property: for ANY weights, the reflected section (own panel solve) gives -CL, -CM and the same CD."""
    net = random_net(seed=seed, scale=3.0)               # strongly non-linear random network
    y = predict_tensors(net, packed([[m, p, t, 10 ** lre, alpha]])).mean(0)[0]
    yr = predict_tensors(net, packed([[-m, p, t, 10 ** lre, -alpha]])).mean(0)[0]
    assert abs(y[0] + yr[0]) < 1e-5 and abs(y[1] - yr[1]) < 1e-5 and abs(y[2] + yr[2]) < 1e-5


def test_check_reflection_passes_for_symmetric_variants_and_fails_otherwise():
    for variant in ("full", "no_panel", "no_drag_prior"):
        assert check_reflection(random_net(variant, scale=3.0), n=64)["ok"]
    for variant in ("plain", "no_symmetry"):
        assert not check_reflection(random_net(variant, scale=3.0), n=64)["ok"]


def test_check_reflection_detects_the_v1_odd_gate(monkeypatch):
    """Regression test: re-introducing the original odd gate tanh(m/0.01) must be caught by the self-test."""
    original = model_mod.make_inputs

    def odd_gate_inputs(m, p, t, lnre, alpha, const, variant="full"):
        x, xm, base = original(m, p, t, lnre, alpha, const, variant)
        pe = 0.45 + (p - 0.45) * torch.tanh(m / 0.01)
        x, xm = x.clone(), xm.clone()
        x[..., 1] = xm[..., 1] = (pe - 0.45) / 0.25
        return x, xm, base

    monkeypatch.setattr(model_mod, "make_inputs", odd_gate_inputs)
    assert not check_reflection(random_net(scale=3.0), n=64)["ok"]


def test_symmetric_section_at_zero_incidence_has_zero_lift_and_moment():
    y = predict_tensors(random_net(scale=3.0), packed([[0.0, 0.0, 0.12, 1e6, 0.0]])).mean(0)[0]
    assert abs(y[0]) < 1e-6 and abs(y[2]) < 1e-6


def test_drag_prior_is_a_plausible_flat_plate_drag():
    cd = torch.exp(drag_prior(torch.log(torch.tensor([1e5, 1e6, 1e7])), torch.tensor([0.12] * 3)))
    assert torch.all(cd[:-1] > cd[1:])                                  # decreases with Re
    assert 0.004 < float(cd[1]) < 0.012                                 # ~ 2 Cf (1 + 2t + 60 t^4) at Re 1e6


def test_ensemble_members_are_independent():
    net = random_net(K=4)
    y = predict_tensors(net, packed([[0.02, 0.4, 0.12, 1e6, 3.0]]))
    assert y.shape == (4, 1, 3)
    assert np.unique(np.round(y[:, 0, 0], 8)).size == 4


def test_save_load_round_trip_is_exact(tmp_path):
    net = random_net(K=2)
    path = tmp_path / "w.npz"
    save_model(net, path, {"note": "hello", "array": np.arange(3)})
    net2, meta = load_model(path)
    assert meta == {"note": "hello", "array": [0, 1, 2]}
    d = packed([[0.03, 0.5, 0.1, 3e5, 7.0]])
    np.testing.assert_array_equal(predict_tensors(net, d), predict_tensors(net2, d))
    assert net2.variant == "full" and net2.K == 2 and net2.width == 32


def test_weights_file_contains_no_pickle(tmp_path):
    path = tmp_path / "w.npz"
    save_model(random_net(), path)
    with np.load(path, allow_pickle=False) as z:        # would raise if any object array were present
        assert all(z[k].dtype != object for k in z.files)


@pytest.mark.parametrize(("field", "value"), [("features_version", FEATURES_VERSION - 1), ("format", "other/0")])
def test_incompatible_weights_are_refused(tmp_path, field, value):
    path = tmp_path / "w.npz"
    save_model(random_net(), path)
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    header = json.loads(str(arrays["header"]))
    header[field] = value
    arrays["header"] = np.array(json.dumps(header))
    with open(path, "wb") as fh:
        np.savez(fh, **arrays)
    with pytest.raises(IncompatibleWeightsError):
        load_model(path)


def test_files_without_header_are_refused(tmp_path):
    path = tmp_path / "x.npz"
    with open(path, "wb") as fh:
        np.savez(fh, a=np.zeros(3))
    with pytest.raises(IncompatibleWeightsError):
        load_model(path)


def test_save_model_keeps_the_exact_file_name(tmp_path):
    path = tmp_path / "model.weights"
    save_model(random_net(), path)
    assert path.exists()


def test_symmetric_forward_requires_reflected_features():
    net = random_net()
    with pytest.raises(ValueError):
        net(torch.zeros(3, 1, 7), None, torch.zeros(3, 1, 3))
