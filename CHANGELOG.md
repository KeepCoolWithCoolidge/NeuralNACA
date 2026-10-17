# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [semantic versioning](https://semver.org/).
Changes that alter model predictions bump the weights' *feature version* and are always called out.

## [1.0.0] - 2026-10-16

First public release.

### Added
- `neuralnaca` package: prediction API (`get_aero`, `NeuralNACA`), shipped 5-member ensemble trained on 4,096 XFOIL
  runs (fingerprint `5988ce08779e0f31`), training-support flag and ensemble spread with every prediction.
- Differentiable Hess-Smith panel method, exactly mirror-symmetric architecture, flat-plate drag prior.
- Robust XFOIL batch driver (python-module and executable backends, crash containment, cold-start and
  NeuralFoil-sweep protocols, Kulfan geometry), self-describing data sets with provenance.
- Benchmark pipeline (`neuralnaca bench`): five replications against all NeuralFoil sizes, training-seed variance,
  ablations (including a physics-only baseline and autograd PINN residuals), data-efficiency curve (512-8,192 runs),
  ground-truth sensitivity (cold vs. NeuralFoil's sweep protocol; exact vs. Kulfan geometry, with a frame-only
  control), support and ensemble-spread study, speed comparison; generated tables and figures.
- `neuralnaca reproduce` with three verification levels; SHA-256 manifest of all shipped files.
- Scripts that rebuild the benchmark's XFOIL builds (verified to reproduce the shipped data bit for bit), and
  `neuralnaca xfoil-check`, which detects broken builds such as the Debian/Ubuntu package.
- Documentation site and test suite (unit, property-based, physics validation, end-to-end, integrity of shipped
  results and documentation numbers).

### Fixed (relative to the pre-release script)
- The camber-position gate was an odd function of camber (`tanh(m/0.01)`), which broke the claimed exact reflection
  symmetry for negative camber. It is now even (`1 - exp(-(m/0.01)^2)`) and checked on every run (feature version 2;
  version-1 weights are refused).
- `OneCycleLR` failed for training runs shorter than ~40 steps.
- NumPy arrays in model metadata could not be serialised.
- XFOIL worker processes that could not start (no `__main__` guard) silently produced all-NaN data; this is now an
  explicit error.
