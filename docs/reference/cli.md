# Command line

All functionality is available through `neuralnaca <command>` (or `python -m neuralnaca <command>`). Global options:
`--version`, `-v/--verbose` (debug logging), `--threads N` (PyTorch CPU threads). Every command has `--help`.

| command | purpose | needs |
|---|---|---|
| `predict` | polar from the shipped (or a given) model | - |
| `info` | version, shipped model fingerprint, domain, environment, XFOIL availability | - |
| `xfoil-check` | does your XFOIL reproduce the benchmark's build? | XFOIL |
| `generate` | sample a design and run XFOIL on it | XFOIL |
| `train` | train a model (and its support model) on a data set | - |
| `evaluate` | score a model on a test set, optionally against NeuralFoil | (neuralfoil) |
| `bench` | run one benchmark experiment into a benchmark root | data or XFOIL |
| `reproduce` | re-derive the shipped results (levels 0, 1, 2) | see [Reproducing](../reproducibility.md) |
| `report` | regenerate `docs/results/tables.md` from the result files | - |
| `figures` | regenerate the documentation figures | matplotlib |

## predict

```text
neuralnaca predict AIRFOIL --re RE [--alpha A0 A1 | --alpha A B C ...] [--step S]
                   [--weights FILE] [--compare-xfoil] [--json]
```

Two `--alpha` values give a range (with `--step`, default 1 degree); any other count is a list of angles. Columns:
`CL`, `CD`, `CM`, the ensemble spread of `CL` and of drag (in %). Conditions outside the domain are marked.

## generate

```text
neuralnaca generate --n N --out FILE [--kind sobol|random] [--seed S] [--backend auto|module|binary]
                    [--executable PATH] [--panels 279] [--protocol cold|nf-sweep] [--geometry naca|kulfan]
                    [--max-iter 100] [--workers W]
```

## train / evaluate

```text
neuralnaca train --data FILE --out WEIGHTS [--ensemble 5] [--steps 18000] [--seed 0]
                 [--variant full|no_symmetry|no_panel|no_drag_prior|plain]
neuralnaca evaluate --weights WEIGHTS --data FILE [--neuralfoil xxxlarge[,size...]] [--out JSON]
```

## bench

```text
neuralnaca bench {main,variance,ablation,curve,protocol,support,speed} --seeds 0-4 [--root .] [--workers W] [--retrain]
                 [--residuals] [--variants a,b,...] [--sizes 512,1024,...]
```

| experiment | what it does | writes |
|---|---|---|
| `main` | data (if absent) → train (if absent) → evaluate vs every NeuralFoil size | `main_s<seed>.json`, `main_s<seed>_pred.npz` |
| `variance` | retrain on the same data with training seeds 101-104 | `variance_s<seed>.json` |
| `ablation` | train each model variant on the same data | `ablation_s<seed>.json` |
| `curve` | train on Sobol prefixes of the training set | `curve_s<seed>.json` |
| `protocol` | re-solve the test set under four ground-truth definitions | `protocol_s<seed>.json` |
| `support` | evaluate the support flag and the ensemble spread | `support.json` |
| `speed` | time NeuralNACA, NeuralFoil and XFOIL per condition (run on an idle machine) | `speed.json` |

The full set of experiments behind the documentation, in order:

```bash
neuralnaca bench main --seeds 0-4
neuralnaca bench protocol --seeds 0
neuralnaca bench variance --seeds 0
neuralnaca bench ablation --seeds 0-2 --variants plain,no_panel,no_symmetry,no_drag_prior,full
neuralnaca bench curve --seeds 0-2 --sizes 512,1024,2048,4096
neuralnaca bench support --seeds 0-4
neuralnaca bench ablation --seeds 0 --residuals --variants full
neuralnaca bench curve --seeds 0 --sizes 8192
neuralnaca bench speed                     # on an otherwise idle machine
neuralnaca report && neuralnaca figures
```

## reproduce

```text
neuralnaca reproduce [--level 0|1|2] [--seeds 0-4] [--root .] [--out DIR] [--json FILE]
```

Exit status 0 if every replication reproduces within tolerance and the central claim holds, 1 otherwise. See
[Reproducing the results](../reproducibility.md).
