# File formats

All files are NumPy `.npz` archives readable with `numpy.load(path, allow_pickle=False)`; none contains pickled
objects, so loading a file from an untrusted source cannot execute code. Metadata is stored as a JSON string.

## XFOIL data sets (`data/xfoil/*.npz`)

| array | shape | content |
|---|---|---|
| `points` | (n, 5) | `m, p, t, Re, alpha` (fractions of chord, degrees) |
| `results` | (n, 3) | XFOIL `CL, CD, CM`; **NaN rows where XFOIL did not converge** (kept, so the design is complete) |
| `meta` | () str | JSON: `design` (n, kind, seed, domain), `xfoil` (configuration + build), `created_utc`, `platform` |

File names: `{train|test}_{n}_s{seed}_{tag}.npz`, where `tag` is `p279_nc9` for the default ground truth and adds
`_{protocol}_{geometry}_it{max_iter}` otherwise. Load with `neuralnaca.data.XfoilDataset.load`.

## Weights (`*.npz` written by `save_model`)

| array | content |
|---|---|
| `header` | JSON: `format` (`neuralnaca-weights/1`), `features_version` (2), `variant`, `K`, `din`, `width`, `depth`, `meta` |
| `W0 ... W4`, `b0 ... b4` | layer weights `(K, d_in, d_out)` and biases `(K, 1, d_out)`, float32 |

Files with another format or feature version are refused (`IncompatibleWeightsError`). `NeuralNACA.fingerprint()`
hashes the parameters (the shipped model is `5988ce08779e0f31`).

## Support model (`*.support.npz`)

| array | content |
|---|---|
| `header` | JSON: `format` (`neuralnaca-support/1`), `cd_max`, `k`, `threshold`, `eps` |
| `points` | training conditions (converged and not) |
| `converged` | bool per training condition |

## Result files (`benchmarks/results/`)

* `main_s<seed>.json`: data set and model records (file names, SHA-256, sizes), training configuration,
  reflection-check result, NeuralFoil version, the full comparison (metrics per model, MAE/RMSE ratios with
  bootstrap intervals, Wilcoxon p-values, win fractions, regime breakdowns) and the software environment.
* `main_s<seed>_pred.npz`: per-point test conditions, XFOIL targets and the predictions of NeuralNACA and
  NeuralFoil-xxxlarge (for pooled analyses and figures).
* `variance_s0.json`, `ablation_s<seed>.json`, `curve_s<seed>.json`, `protocol_s0.json`, `support.json`: the studies
  described in [Results](../results.md).
* `polars_reference.npz`: XFOIL and NeuralFoil polars used by the figures.

`benchmarks/MANIFEST.json` lists the SHA-256 of every shipped data, model and result file;
`neuralnaca.data.verify_manifest` checks them.
