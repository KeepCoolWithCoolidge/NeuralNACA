"""The public prediction API, including physical plausibility of the shipped model across its domain."""

import numpy as np
import pytest

import neuralnaca
from neuralnaca import DEFAULT_DOMAIN, DomainWarning, NeuralNACA, UnsupportedWarning, get_aero

#: SHA-256 prefix of the shipped weights. Changing the shipped model must be a deliberate act.
SHIPPED_FINGERPRINT = "5988ce08779e0f31"


def test_shipped_model_identity(pretrained):
    assert pretrained.fingerprint() == SHIPPED_FINGERPRINT
    assert pretrained.ensemble_size == 5 and pretrained.net.variant == "full"
    assert pretrained.meta["n_train_runs"] == 4096
    assert "NeuralNACA(" in repr(pretrained)


def test_scalar_inputs_give_scalar_outputs(pretrained):
    out = pretrained.predict("2412", alpha=4.0, Re=1e6)
    assert set(out) == {"CL", "CD", "CM", "CL_std", "lnCD_std", "CD_std", "CM_std", "in_domain", "xfoil_convergence",
                        "supported"}
    assert all(np.shape(v) == () for v in out.values())
    assert 0.6 < out["CL"] < 0.8 and 0.005 < out["CD"] < 0.01 and -0.07 < out["CM"] < -0.04


def test_broadcasting(pretrained):
    out = pretrained.predict(alpha=np.linspace(-4, 12, 5)[:, None], Re=np.array([2e5, 1e6, 5e6]), m=0.02, p=0.4,
                             t=0.12)
    assert out["CL"].shape == (5, 3)
    out = pretrained.predict(m=np.array([0.0, 0.02, 0.04]), p=0.4, t=0.12, alpha=2.0, Re=1e6, return_members=True)
    assert out["CL"].shape == (3,) and out["members"].shape == (5, 3, 3)
    assert np.all(np.diff(out["CL"]) > 0)                              # more camber -> more lift


def test_designation_and_parameters_are_equivalent(pretrained):
    a = pretrained.predict("NACA 4415", alpha=[0, 6], Re=7e5)
    b = pretrained.predict(m=0.04, p=0.4, t=0.15, alpha=[0, 6], Re=7e5)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def test_symmetric_section_symmetry(pretrained):
    a = np.array([-5.0, -2.0, 0.0, 3.0, 5.0])
    out = pretrained.predict("0012", alpha=a, Re=1e6)
    flip = pretrained.predict("0012", alpha=-a, Re=1e6)
    np.testing.assert_allclose(out["CL"], -flip["CL"], atol=1e-6)
    np.testing.assert_allclose(out["CM"], -flip["CM"], atol=1e-6)
    np.testing.assert_allclose(out["CD"], flip["CD"], rtol=1e-6)
    assert abs(out["CL"][2]) < 1e-6


def test_camber_position_ignored_for_symmetric_sections(pretrained):
    a = pretrained.predict(m=0.0, p=0.25, t=0.1, alpha=3.0, Re=1e6)
    b = pretrained.predict(m=0.0, p=0.65, t=0.1, alpha=3.0, Re=1e6)
    assert a["CL"] == b["CL"] and a["CD"] == b["CD"]


def test_domain_handling(pretrained):
    with pytest.warns(DomainWarning, match="outside the validated domain"):
        out = pretrained.predict("2412", alpha=[0.0, 20.0], Re=1e6)
    np.testing.assert_array_equal(out["in_domain"], [True, False])
    with pytest.raises(ValueError, match="outside"):
        pretrained.predict("9412", alpha=0.0, Re=1e6, on_domain_error="raise")
    out = pretrained.predict("2412", alpha=0.0, Re=5e4, on_domain_error="ignore")
    assert not out["in_domain"]


@pytest.mark.parametrize("kwargs", [
    {"airfoil": "2412", "m": 0.02},
    {"m": 0.02, "p": 0.4},
    {"m": 0.02, "p": 0.0, "t": 0.12},
    {"m": 0.02, "p": 0.4, "t": -0.12},
    {"m": 0.02, "p": 0.4, "t": 0.12, "Re": -1.0},
    {"m": np.nan, "p": 0.4, "t": 0.12},
    {"airfoil": "2412", "on_domain_error": "maybe"},
])
def test_invalid_inputs(pretrained, kwargs):
    kw = {"alpha": 2.0, "Re": 1e6} | kwargs
    with pytest.raises(ValueError):
        pretrained.predict(**kw)


def test_panel_cache_is_bounded(pretrained, monkeypatch):
    import neuralnaca.api as api

    monkeypatch.setattr(api, "_CACHE_LIMIT", 10)
    m = NeuralNACA(pretrained.net, pretrained.meta)
    for k in range(4):
        m.predict(m=np.linspace(0.01, 0.05, 4) + k * 1e-3, p=0.4, t=0.12, alpha=2.0, Re=1e6)
    assert len(m._const_cache) <= 10


def test_get_aero_uses_the_shipped_model(pretrained):
    a = get_aero("2412", alpha=[1.0, 2.0], Re=3e5)
    b = pretrained.predict("2412", alpha=[1.0, 2.0], Re=3e5)
    np.testing.assert_array_equal(a["CL"], b["CL"])


def test_polar_helper(pretrained):
    pol = pretrained.polar("2412", Re=1e6)
    assert pol["alpha"][0] == DEFAULT_DOMAIN.alpha[0] and pol["alpha"][-1] == DEFAULT_DOMAIN.alpha[1]
    assert pol["CL"].shape == pol["alpha"].shape


def test_load_from_file(tmp_path, pretrained):
    from neuralnaca.model import save_model

    save_model(pretrained.net, tmp_path / "w.npz", {"x": 1})
    m = NeuralNACA.load(tmp_path / "w.npz")
    assert m.fingerprint() == pretrained.fingerprint() and m.meta == {"x": 1} and m.support is None
    pretrained.support.save(tmp_path / "w.support.npz")
    assert NeuralNACA.load(tmp_path / "w.npz").support is not None                    # auto-detected sibling
    assert NeuralNACA.load(tmp_path / "w.npz", support=None).support is None


def test_uncertainty_outputs_are_consistent(pretrained):
    out = pretrained.predict("4412", alpha=np.arange(-5, 16), Re=4e5)
    assert (out["CL_std"] >= 0).all() and (out["lnCD_std"] >= 0).all()
    np.testing.assert_allclose(out["CD_std"], out["CD"] * out["lnCD_std"])


def test_public_namespace():
    for name in neuralnaca.__all__:
        assert hasattr(neuralnaca, name)
    assert neuralnaca.__version__.count(".") == 2


# ------------------------------------------------------------------------------------------------ plausibility
GRID_SECTIONS = ["0006", "0012", "0018", "2412", "4412", "4418", "6409", "8412", "2306", "3712", "5615"]
GRID_RE = [1e5, 3e5, 1e6, 3e6, 1e7]


def test_physical_plausibility_across_the_domain(pretrained):
    """Coarse physics checks of the shipped model on a grid that spans the whole domain (supported predictions)."""
    a = np.arange(-5.0, 15.01, 0.5)
    n_sup = n_tot = 0
    for code in GRID_SECTIONS:
        for re in GRID_RE:
            out = pretrained.predict(code, alpha=a, Re=re, on_domain_error="ignore")
            cl, cd, cm, sup = out["CL"], out["CD"], out["CM"], out["supported"]
            n_sup, n_tot = n_sup + sup.sum(), n_tot + sup.size
            assert np.isfinite(cl).all() and np.isfinite(cm).all() and (cd > 0).all()
            assert ((cd[sup] > 0.003) & (cd[sup] < 0.16)).all(), (code, re)
            lin = (a >= -2) & (a <= 4)                                  # attached-flow range: always supported
            assert sup[lin].all(), (code, re)
            if re >= 5e5:
                # thin-airfoil value 2 pi / rad = 0.110 / deg. (At Re ~ 1e5 XFOIL's lift curve itself is not linear
                # near zero lift - laminar separation steepens it locally to ~0.27 / deg for NACA 0012 - so the band
                # is only applied where the boundary layer is mostly attached.)
                slope = np.polyfit(a[lin], cl[lin], 1)[0]
                assert 0.07 < slope < 0.13, (code, re, slope)
            assert (np.diff(cl[(a >= -4) & (a <= 6)]) > 0).all(), (code, re)
            assert np.abs(cm[sup]).max() < 0.25
    assert n_sup / n_tot > 0.9


def test_unsupported_corner_is_flagged(pretrained):
    """Deep stall of thin sections at low Re: no XFOIL convergence in training -> flagged, with a warning."""
    with pytest.warns(UnsupportedWarning, match="without training support"):
        out = pretrained.predict("0006", alpha=[2.0, 15.0], Re=1e5)
    np.testing.assert_array_equal(out["in_domain"], [True, True])
    np.testing.assert_array_equal(out["supported"], [True, False])
    assert out["xfoil_convergence"][0] > 0.8 and out["xfoil_convergence"][1] < 0.2
    with pytest.raises(ValueError, match="training support"):
        pretrained.predict("0006", alpha=15.0, Re=1e5, on_domain_error="raise")


def test_without_support_model_supported_equals_in_domain(pretrained):
    bare = NeuralNACA(pretrained.net, pretrained.meta)
    out = bare.predict("0006", alpha=[2.0, 15.0], Re=1e5)
    np.testing.assert_array_equal(out["supported"], out["in_domain"])
    assert np.isnan(out["xfoil_convergence"]).all()


def test_drag_falls_with_reynolds_number_in_attached_flow(pretrained):
    for code in ("0012", "2412", "4415"):
        cd = pretrained.predict(code, alpha=2.0, Re=np.array(GRID_RE))["CD"]
        assert cd[0] > cd[-1] and cd[1] > cd[-1]
