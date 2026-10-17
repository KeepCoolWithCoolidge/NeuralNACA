# NeuralNACA

**A physics-embedded neural surrogate for NACA 4-digit airfoil polars that matches XFOIL more closely than
NeuralFoil, trained on 0.052 % of NeuralFoil's XFOIL runs.**

[![CI](https://github.com/KeepCoolWithCoolidge/NeuralNACA/actions/workflows/ci.yml/badge.svg)](https://github.com/KeepCoolWithCoolidge/NeuralNACA/actions/workflows/ci.yml)
[![Docs](https://github.com/KeepCoolWithCoolidge/NeuralNACA/actions/workflows/docs.yml/badge.svg)](https://keepcoolwithcoolidge.github.io/NeuralNACA)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

```python
import neuralnaca
aero = neuralnaca.get_aero("NACA 2412", alpha=4.0, Re=1e6)   # CL, CD, CM + uncertainty and support flags
```

NeuralNACA predicts what XFOIL 6.99 computes (free transition, N<sub>crit</sub> = 9, Mach 0) for NACA 4-digit sections
with 0-8 % camber, 6-18 % thickness, Re 10<sup>5</sup>-10<sup>7</sup> and -5° to 15° incidence. Physics enters in
three ways:

- a differentiable Hess-Smith panel method supplies the inviscid lift and moment;
- the architecture enforces exact reflection symmetry;
- a flat-plate skin-friction prior anchors the drag.

A five-member neural ensemble learns the viscous correction from 4,096 XFOIL runs.

## Result

5 independent replications (each: a fresh training set of 4,096 XFOIL runs and a fresh test set of unseen airfoils),
13,933 held-out test points, scored against XFOIL. The ratio is NeuralNACA / NeuralFoil-xxxlarge
(NeuralFoil 0.3.3) with a 95 % paired-bootstrap interval; below 1 means NeuralNACA is closer to XFOIL.

| output | MAE ratio | RMSE ratio | median abs. error, NeuralNACA vs NeuralFoil |
|---|---|---|---|
| C<sub>L</sub> | **0.62 [0.58, 0.65]** | 0.97 [0.82, 1.15] | 0.00080 vs 0.00236 |
| ln C<sub>D</sub> | **0.65 [0.62, 0.68]** | 0.88 [0.80, 0.97] | 0.00306 vs 0.00717 |
| C<sub>M</sub> | **0.60 [0.57, 0.64]** | 0.99 [0.82, 1.21] | 0.000152 vs 0.000444 |

- **Lower mean error in every replication.** The MAE ratio's interval is below 1 for all three outputs in each of
  the 5 replications, including the two generated after the code was frozen.
- **Smaller typical error.** Median errors are 2.3-3x smaller.
- **No gain on the largest errors.** NeuralNACA is not better on its worst cases: RMSE is a statistical tie for
  C<sub>L</sub> and C<sub>M</sub>, because of rare large errors in a sparsely sampled corner (high camber, low Re,
  negative incidence; see [Limitations](https://keepcoolwithcoolidge.github.io/NeuralNACA/limitations/)).
- **It beats every NeuralFoil size,** not only the largest.
- **Why it wins: mostly geometry, not a better network.** NeuralFoil analyses an 8+8-weight Kulfan fit of the section,
  not the exact NACA shape. The fit is geometrically excellent, but XFOIL's answer for the fitted shape differs from
  that for the exact section by about as much as NeuralFoil's entire error. NeuralFoil's error against XFOIL on its
  *own* fitted geometry is smaller than NeuralNACA's against XFOIL on the exact geometry. Even a plain MLP without
  physics, trained on 4,096 exact-geometry runs, beats NeuralFoil here.
- **What the physics adds.** Without the panel-method backbone the error grows by 21-41 %. Exact symmetry and the
  drag prior add consistency, not accuracy. Break-even with NeuralFoil is at about 2,000 XFOIL runs.

Details, ablations and the ground-truth study are on the [results page](https://keepcoolwithcoolidge.github.io/NeuralNACA/results/); the raw generated tables
are in [docs/results/tables.md](docs/results/tables.md).

![polars](docs/assets/polars_re1e06.png)

**Scope, stated plainly.** This is a *specialist*: one airfoil family, one flow model, a bounded domain. NeuralFoil is
a *generalist* (any airfoil, any incidence, a far wider range of Re and transition settings). For NACA 4-digit
sections, exact geometry plus specialisation beats NeuralFoil with very little data. That does not make NeuralNACA the
better general tool, and the errors are relative to XFOIL, not experiments.

## Install

```bash
pip install git+https://github.com/KeepCoolWithCoolidge/NeuralNACA              # predictions (NumPy, SciPy, PyTorch)
pip install "neuralnaca[neuralfoil,plot] @ git+https://github.com/KeepCoolWithCoolidge/NeuralNACA"   # + comparisons and figures
```

(Once published on PyPI: `pip install neuralnaca`.)

XFOIL is only needed to generate new data; the [installation guide](https://keepcoolwithcoolidge.github.io/NeuralNACA/installation/) has verified build scripts. The
Debian/Ubuntu `xfoil` package is broken; `neuralnaca xfoil-check` detects that.

## Use

```python
from neuralnaca import NeuralNACA

model = NeuralNACA.pretrained()
out = model.predict("4412", alpha=[0.0, 4.0, 8.0], Re=3e5)
out["CL"], out["CD"], out["CM"]               # ensemble means
out["CL_std"], out["supported"]               # how much to trust each prediction
```

```console
$ neuralnaca predict 2412 --re 1e6 --alpha -4 12 --step 2
$ neuralnaca predict 2412 --re 1e6 --compare-xfoil        # side by side with XFOIL
```

Every prediction says whether it is backed by training data. Where XFOIL itself does not converge (deep stall of thin
sections at low Re), the model has no data and its output can be meaningless. Such conditions are flagged
(`supported = False`, `UnsupportedWarning`).

## Reproduce

```bash
git clone https://github.com/KeepCoolWithCoolidge/NeuralNACA && cd NeuralNACA && pip install -e ".[neuralfoil,plot,test]"
neuralnaca reproduce --level 0     # shipped weights + shipped XFOIL data -> every published number (minutes)
neuralnaca reproduce --level 1     # retrain all 5 replications from the shipped data (~30 min, no XFOIL)
neuralnaca reproduce --level 2     # regenerate the XFOIL data too
pytest                             # 200+ tests, incl. integrity of all shipped results and doc numbers
```

All data, models and results are in the repository with SHA-256 checksums. Every number in the documentation is
generated from them, and the tests fail if any of them drift apart. See
[Reproducing the results](https://keepcoolwithcoolidge.github.io/NeuralNACA/reproducibility/).

## Documentation

Full documentation: **https://keepcoolwithcoolidge.github.io/NeuralNACA** (sources in [docs/](docs); build locally with `mkdocs serve`).

* [Method](https://keepcoolwithcoolidge.github.io/NeuralNACA/method/): panel method, symmetry proof, features, training, support estimate
* [Benchmark protocol](https://keepcoolwithcoolidge.github.io/NeuralNACA/benchmark/): ground truth, designs, statistics, threats to validity
* [Results](https://keepcoolwithcoolidge.github.io/NeuralNACA/results/): every table and figure, ablations, data efficiency, ground-truth sensitivity
* [Limitations](https://keepcoolwithcoolidge.github.io/NeuralNACA/limitations/): where not to trust it, and ideas that did not work
* [User guide](https://keepcoolwithcoolidge.github.io/NeuralNACA/guide/predictions/), [API](https://keepcoolwithcoolidge.github.io/NeuralNACA/reference/api/), [CLI](https://keepcoolwithcoolidge.github.io/NeuralNACA/reference/cli/)

## Citation

See [CITATION.cff](CITATION.cff). Please also cite XFOIL (Drela, 1989) and NeuralFoil (Sharpe & Hansman,
arXiv:2503.16323), on whose work this comparison rests.

## License

MIT. XFOIL (GPL) is not distributed with this package.
