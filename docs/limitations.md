# Limitations

This page lists, as concretely as we can, where NeuralNACA should **not** be trusted, and what the benchmark does
**not** show. Most items were found by deliberately looking for failures; several are covered by tests.

## Scope

* **One family, one box.** NACA 4-digit sections with \(0 \le m \le 0.08\), \(0.2 \le p \le 0.7\),
  \(0.06 \le t \le 0.18\), \(10^5 \le \mathrm{Re} \le 10^7\), \(-5^\circ \le \alpha \le 15^\circ\). Nothing is claimed
  outside this box. NACA 5-digit, 6-series and arbitrary sections are not supported.
* **One flow model.** Free transition with \(N_\mathrm{crit} = 9\), Mach 0, no forced trips, no flaps, two-dimensional.
  NeuralFoil covers all of these; NeuralNACA covers none of them.
* **A specialist against a generalist.** NeuralFoil was trained for arbitrary airfoils across a much larger flow
  domain. Beating it on its home ground with a small fraction of the data is a statement about what a specialist can
  do in a narrow family, not about which tool is better in general.

## What the advantage is, and what it is not

NeuralNACA predicts XFOIL for the *exact* NACA section; NeuralFoil predicts XFOIL for its own 8+8-weight Kulfan fit of
it. The [ground-truth study](results.md#sensitivity-to-the-definition-of-the-xfoil-ground-truth) shows that the two
XFOIL answers differ by about as much as NeuralFoil's whole error. A control with the exact shape in NeuralFoil's
coordinate frame shows that the frame transformation contributes almost nothing to \(C_L\) and \(C_D\), so the
shape fit is the cause. Against XFOIL on its own fitted geometry, NeuralFoil is the more accurate model.

So the benchmark establishes that **for NACA 4-digit sections, NeuralNACA is closer to XFOIL than NeuralFoil**,
mainly because it represents the geometry exactly and is specialised to it. It does **not** establish that NeuralNACA
is a more faithful XFOIL emulator in general, or that physics embedding beats a large data set. A plain network
trained on the same exact-geometry data also beats NeuralFoil here (see the ablation).

## The ground truth is XFOIL, not reality

Every error reported here is an error with respect to XFOIL 6.99. XFOIL's own errors are inherited unchanged. They
grow near and beyond maximum lift, at low Reynolds number with long laminar separation bubbles, and for thick
sections at high incidence. Nothing in this project validates against wind-tunnel data.

## Where XFOIL does not converge, the model is unsupported, and can be absurd

About 7 % of random conditions in the domain do not converge in XFOIL. They concentrate in deep stall of thin
sections at low Reynolds number. For example, none of the 27 points of the shipped model's training set with
\(t < 0.08\), \(\mathrm{Re} < 2\times10^5\) and \(\alpha > 10^\circ\) converged. The model never saw data there, and its drag extrapolates without bound. NACA 0006 at
Re = 10⁵ and 15° yields \(C_D \approx 6.6\), which is physically meaningless.

These conditions **cannot be scored** (XFOIL has no answer), so the benchmark is silent about them. Instead,
NeuralNACA *flags* them. `supported` is false and an `UnsupportedWarning` is raised when the estimated XFOIL
convergence probability is below 0.5 or the predicted drag exceeds the largest drag in the training data. How well
this flag works on held-out data is reported in [Results](results.md#knowing-when-not-to-trust-a-prediction). It does
not catch every non-converged condition, because XFOIL's failures are partly random. A test pins the behaviour for the
known-bad corner.

## Sparse corners with sharp physics: rare but large errors

The advantage over NeuralFoil is an advantage *on average and in the typical case*. Among the worst individual errors
of NeuralNACA, one pattern recurs. Highly cambered sections (\(m \approx 0.07\)-\(0.08\)) at low Reynolds number
(1-2 × 10⁵) and negative incidence develop lower-surface separation. XFOIL converges there, but 4,096 training runs
sample this corner sparsely. In one replication the five worst points account for more than half of NeuralNACA's
squared \(C_L\) error (a third for NeuralFoil), and errors in \(C_L\) of up to 0.37 occur where NeuralFoil is within
0.01.

**Consequence:** RMSE, which is dominated by the largest errors, does **not** consistently favour NeuralNACA. In some
replications the \(C_L\) or \(C_M\) RMSE ratio is above 1 (see the per-replication table in
[Results](results.md#per-replication)). If your application cares about worst-case rather than typical error in that
corner, NeuralFoil is the more robust choice there, or NeuralNACA needs more training data in it.

**Mitigation:** the ensemble spread is a good indicator of these errors. It correlates with the actual error, and
most of the largest errors have a spread in the top 5 % (statistics in
[Results](results.md#knowing-when-not-to-trust-a-prediction)). Check `CL_std` / `CM_std`.

## Low-Reynolds-number behaviour

At \(\mathrm{Re} \approx 10^5\) XFOIL's polars are strongly non-linear even at small incidence. NACA 0012's lift jumps
from 0 to 0.26 between 0° and 1° because of laminar separation. NeuralNACA reproduces this but slightly smooths it.
Plausibility tests therefore apply the thin-airfoil lift-slope band only at \(\mathrm{Re} \ge 5\times10^5\).

## Uncertainty outputs are heuristic

The ensemble standard deviations rank errors well but are not calibrated confidence intervals: five members are too
few, and they share the same training data, so they can agree and still be wrong in the same direction.

## The panel method is a feature extractor, not a converged potential-flow solution

With the open NACA trailing edge, the discrete Kutta condition makes the 120-panel solution drift slowly under
refinement instead of converging (details in [Method](method.md#2-inviscid-baseline-a-differentiable-hess-smith-panel-method)).
At the fixed resolution used it is within 2 % of XFOIL's inviscid lift, and the network corrects the rest. Do not use
`neuralnaca.panel` as a reference inviscid solver for open-trailing-edge sections without closing the gap.

## Dependence on how the ground truth is defined

XFOIL's answer depends on the solution protocol (cold start vs. warm-started sweeps) and on the exact geometry (NACA
coordinates vs. NeuralFoil's Kulfan fit), mostly near stall. NeuralNACA was trained on cold-start labels of the exact
geometry, the setting it is scored on. The [sensitivity study](results.md#sensitivity-to-the-definition-of-the-xfoil-ground-truth)
re-scores both models under five definitions, including NeuralFoil's own.

## Reproducibility across machines

Results are bit-for-bit reproducible on the same platform and software versions (verified). On other CPUs, BLAS
libraries or PyTorch versions, floating-point summation order can differ, and training amplifies such differences.
Retrained models then differ slightly. The size of this effect is bounded by the training-seed study and is what the
level-1 reproduction tolerance is based on. XFOIL builds other than the documented one can differ near stall; use
`neuralnaca xfoil-check`.

## Things we tried that did not help

Kept here so that nobody has to rediscover them. All were evaluated on development draws separate from the
benchmark.

* **Autograd PINN residuals** (lift-slope band, curvature bound, drag monotonic in Re) on unlabelled collocation points:
  consistent with XFOIL but no accuracy gain, at several times the training cost (also measured in the ablation).
* **Boundary-layer integral features** (Thwaites/Michel transition estimates, Squire-Young drag): hurt at this data
  size. The features are noisy near the separation boundaries where the error is concentrated.
* **Inviscid \(C_{p,\min}\) and other extra inviscid features:** no gain.
* **Wider networks:** no gain at 4,096 runs.
* **Mirror symmetry and the drag prior** are kept for consistency and robustness, but in the ablation neither
  measurably improves the MAE (changes of at most 4 %). Earlier development runs with single networks suggested a
  small gain from the symmetry; the ensemble benchmark does not confirm it.
* **Training on fewer than about 2,000 runs:** NeuralNACA falls behind NeuralFoil (see the data-efficiency curve).
