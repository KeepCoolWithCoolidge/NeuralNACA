# Benchmark protocol

**Question.** For NACA 4-digit sections inside the [domain](guide/predictions.md#the-validated-domain), is NeuralNACA,
trained on a few thousand XFOIL runs, more accurate *with respect to XFOIL* than NeuralFoil, which was trained on
7.9 million XFOIL runs?

Everything below is implemented in `neuralnaca.benchmark` and executed by `neuralnaca bench ...`. The shipped files in
`data/` and `benchmarks/` are its outputs, and [Reproducing the results](reproducibility.md) re-derives them.

## Ground truth

| setting | value | why |
|---|---|---|
| solver | XFOIL 6.99 (python module, commit `5c8c4ff`, built without floating-point traps) | the program NeuralFoil was trained to imitate |
| geometry | exact NACA 4-digit coordinates (100 per side, cosine spacing) | what a user would hand to either tool |
| panels | 279 (XFOIL `PPAR` re-panelling) | AeroSandbox's default, used for NeuralFoil's data |
| transition | free, \(N_\mathrm{crit} = 9\) | NeuralFoil's default |
| Mach | 0 | |
| iterations | 100 | AeroSandbox's default |
| protocol | cold start for every point | answers independent of run order (see below) |
| scoring | XFOIL-converged points only | there is no reference value otherwise |

The executable backend reproduces the module's results to the four printed decimals at benign conditions. At
conditions where XFOIL has more than one solution, the two backends (like any two protocols) can land on different
branches; this is part of the noise floor quantified below.

## Data and replications

Each **replication** \(s\) (seeds 0-4) consists of:

* a **training set**: 4,096 conditions from a scrambled Sobol design (seed \(s\)), 20 % of them forced to be symmetric
  sections, Re log-uniform, angles rounded to 0.001°;
* a **test set**: 3,000 conditions drawn i.i.d. uniformly (independent generator, seed \(s + 1000\)), so that the
  test error is an unbiased estimate of the domain-averaged error;
* a model trained on the converged training points with the fixed recipe;
* an evaluation on the converged test points.

**No test airfoil occurs in any form in the training set of its replication** (checked in every run, and the run
aborts otherwise). The sections in the documentation figures (0012, 2412, 4412, 6409) occur in no training set at
all. The test suite verifies both properties on the shipped files.

Replications 0-2 were generated during development, but their test sets were never used to choose anything.
Replications 3 and 4 were generated and evaluated after the code freeze. Five replications give an honest picture of
the run-to-run spread caused by the random data draw and the training randomness together; a separate study
isolates the training randomness.

## NeuralFoil configuration

* `neuralfoil.get_aero_from_coordinates(coordinates, alpha, Re, n_crit=9, model_size=...)` with free transition
  (`xtr_upper = xtr_lower = 1`).
* The **same exact NACA coordinates** that XFOIL receives. NeuralFoil fits its 8+8-weight Kulfan parameterisation to
  them internally; that fitting step is part of NeuralFoil's pipeline and therefore of its error. The
  [truth-sensitivity study](#sensitivity-to-the-ground-truth) separates the two.
* All eight model sizes (`xxsmall` ... `xxxlarge`); `xxxlarge`, the largest, is the primary reference.
* Version 0.3.3 (recorded in every result file).

## Metrics and statistics

* The **primary metrics** are NeuralFoil's own: mean absolute error (MAE) of \(C_L\), \(\ln C_D\) and \(C_M\). An error
  of 0.01 in \(\ln C_D\) is about 1 % of drag.
* **Secondary metrics:** RMSE (sensitive to the largest errors), median absolute error (typical case), 95th
  percentile, mean relative drag error, the share of points where one model is closer to XFOIL, and a breakdown by
  angle of attack and Reynolds number.
* **Uncertainty:** the ratio of MAEs (NeuralNACA / NeuralFoil) with a 95 % *paired* bootstrap interval (2,000
  resamples of test points; both models are always scored on the same points). A distribution-free Wilcoxon
  signed-rank test of the paired absolute errors complements it. The bootstrap's coverage is itself tested in the
  test suite.
* **Claim criterion:** NeuralNACA "beats" NeuralFoil on an output if the upper end of the 95 % interval of the MAE
  ratio is below 1. The central claim requires this for all three outputs, in every replication and pooled.

## Training-data accounting

* NeuralNACA: 4,096 XFOIL runs attempted per model (about 93 % converged).
* NeuralFoil: 7,913,292 XFOIL runs attempted, 56 % converged (Sharpe & Hansman, arXiv:2503.16323, Sec. III.E).

Both counts include non-converged runs. The ratio, 0.052 %, compares the XFOIL effort of the two approaches. It does
not compare their scope: NeuralFoil is a generalist (arbitrary airfoils, ±180° incidence, Re from 10² to 10¹⁰,
variable transition settings), NeuralNACA a specialist for one airfoil family in a bounded domain.

## Threats to validity, and how each is addressed

| concern | how it is addressed |
|---|---|
| A specialist is compared with a generalist | Stated up front everywhere; the domain and the family are fixed in advance; the claim is restricted to them. |
| The ground-truth protocol and geometry differ from those NeuralFoil was trained on | [Truth-sensitivity study](#sensitivity-to-the-ground-truth): both models scored against five XFOIL definitions, including NeuralFoil's own protocol on NeuralFoil's own geometry, with a control that separates the frame transformation from the shape fit. |
| Hyper-parameters tuned on the test data | Tuned on separate development draws; replications 3-4 generated after the freeze. |
| One lucky data draw | Five independent replications plus a pooled analysis; per-replication results are all reported. |
| Training randomness | Training-seed study on fixed data; its spread sets the reproduction tolerance. |
| Averages hide failures | Medians, 95th percentiles, RMSE, error distributions, regime breakdowns, and an explicit search for worst cases (see [Limitations](limitations.md)). |
| Only converged points are scored | The share and location of non-converged points are reported; NeuralNACA flags them as unsupported, and the accuracy on supported points is reported separately. |
| XFOIL is not unique near stall | XFOIL-vs-XFOIL differences between protocols bound the attainable accuracy and are reported next to the model errors. |
| Leakage between training and test | Airfoil-level disjointness checked in every run and in the test suite. |
| Software versions | Every result file records Python, NumPy, SciPy, PyTorch, NeuralFoil, AeroSandbox and the XFOIL build. |
| Silent drift between data, models, results and documentation | SHA-256 manifest; every number in the documentation is generated from the result files, and tests fail if they disagree. |

## Sensitivity to the ground truth

The test points of replication 0 are re-solved under five definitions of "what XFOIL says":

| name | protocol | geometry | iterations |
|---|---|---|---|
| `cold/naca` | cold start per point | exact NACA (the benchmark's ground truth) | 100 |
| `cold/naca-normalized` | cold start per point | exact NACA, in NeuralFoil's normalised frame (control) | 100 |
| `cold/kulfan` | cold start per point | NeuralFoil's 8+8 Kulfan fit, in its normalised frame | 100 |
| `nf-sweep/naca` | NeuralFoil's warm-started 5° sweeps from 0° | exact NACA | 200 |
| `nf-sweep/kulfan` | NeuralFoil's warm-started 5° sweeps from 0° | NeuralFoil's Kulfan fit | 200 |

* `nf-sweep/kulfan` is how NeuralFoil's training labels were produced (see `asb.XFoil.alpha(start_at=0)` and
  NeuralFoil's `training/` scripts), so it is the definition most favourable to NeuralFoil.
* Comparing `cold/naca` with `cold/naca-normalized` isolates NeuralFoil's frame transformation (rotation and
  scaling to its normalised frame, angle and Reynolds-number adjustment, moment transfer).
* Comparing `cold/naca-normalized` with `cold/kulfan` isolates the effect of the Kulfan *shape* fit.

Both models are scored on the points that converged under all definitions, and the study reports how much XFOIL
disagrees with itself between definitions. Results:
[Results](results.md#sensitivity-to-the-definition-of-the-xfoil-ground-truth).
