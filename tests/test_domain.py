import numpy as np
import pytest

from neuralnaca.domain import DEFAULT_DOMAIN, Domain, sample_points


@pytest.mark.parametrize("kind", ["sobol", "random"])
def test_sampled_points_lie_in_domain(kind):
    pts = sample_points(2048, kind, seed=3)
    assert pts.shape == (2048, 5)
    assert DEFAULT_DOMAIN.contains(*pts.T).all()
    np.testing.assert_array_equal(np.round(pts[:, 4], 3), pts[:, 4])        # angles rounded to 1e-3 deg


def test_symmetric_fraction_and_log_uniform_reynolds():
    pts = sample_points(4096, "sobol", seed=0)
    assert (pts[:, 0] == 0).mean() == pytest.approx(DEFAULT_DOMAIN.sym_fraction, abs=0.01)
    lr = np.log10(pts[:, 3])
    assert lr.mean() == pytest.approx(6.0, abs=0.02)                            # log-uniform on [5, 7]
    assert np.histogram(lr, bins=4, range=(5, 7))[0].min() > 900


def test_sampling_is_deterministic_and_seed_dependent():
    np.testing.assert_array_equal(sample_points(64, "sobol", 1), sample_points(64, "sobol", 1))
    assert not np.array_equal(sample_points(64, "sobol", 1), sample_points(64, "sobol", 2))
    np.testing.assert_array_equal(sample_points(64, "random", 1), sample_points(64, "random", 1))


def test_sobol_prefix_property():
    """The first 2^k points of a longer Sobol design are the 2^k design: the data-efficiency curve relies on it."""
    np.testing.assert_array_equal(sample_points(8192, "sobol", 0)[:4096], sample_points(4096, "sobol", 0))


def test_invalid_design_arguments():
    with pytest.raises(ValueError):
        sample_points(10, "grid", 0)
    with pytest.raises(ValueError):
        sample_points(0, "sobol", 0)


def test_contains_edges_and_symmetric_sections():
    d = Domain()
    assert d.contains(0.0, 0.0, 0.12, 1e6, 0.0)                    # p irrelevant for m = 0
    assert not d.contains(0.02, 0.0, 0.12, 1e6, 0.0)               # ... but not for cambered sections
    assert d.contains(0.08, 0.7, 0.18, 1e7, 15.0)                  # closed box
    assert d.contains(0.0, 0.0, 0.06, 1e5, -5.0)
    assert not d.contains(0.09, 0.4, 0.12, 1e6, 0.0)
    assert not d.contains(0.02, 0.4, 0.12, 2e7, 0.0)
    assert not d.contains(0.02, 0.4, 0.12, 1e6, 15.5)
    assert not d.contains(-0.02, 0.4, 0.12, 1e6, 0.0)              # reflected sections are not in the domain
    got = d.contains(np.array([0.0, 0.1]), 0.4, 0.12, 1e6, np.array([[0.0], [20.0]]))
    np.testing.assert_array_equal(got, [[True, False], [False, False]])


def test_describe_mentions_every_range():
    s = DEFAULT_DOMAIN.describe()
    for token in ("m in", "p in", "t in", "Re in", "alpha in"):
        assert token in s
