# FAQ

**How can a model trained on 0.05 % of NeuralFoil's data be more accurate?**
Mostly because of geometry, and partly because of specialisation, and **not** because it is a better XFOIL emulator.

* **Geometry.** NeuralFoil does not see the NACA section itself: it analyses an 8+8-weight Kulfan (CST) fit of it.
  The fit is geometrically excellent (shape errors of hundredths of a percent of chord), but XFOIL's transition
  prediction is sensitive enough that the fitted shape gives a noticeably different answer. That difference is about
  as large as NeuralFoil's entire error against the exact section. Scored against XFOIL on its *own* fitted shape,
  NeuralFoil is more accurate than NeuralNACA. Even a plain MLP without physics, trained on 4,096 exact-geometry XFOIL
  runs, beats NeuralFoil on NACA sections.
* **Specialisation.** NeuralFoil must represent every airfoil shape, ±180° of incidence, Reynolds numbers from 10² to
  10¹⁰ and arbitrary transition settings with one network. NeuralNACA covers one three-parameter family in a bounded
  box at one transition setting.
* **Physics.** The panel-method backbone lowers the error by 17-29 % relative to the same model without it.

See the [ground-truth study](results.md#sensitivity-to-the-definition-of-the-xfoil-ground-truth) and the
[ablation](results.md#ablation-what-each-physics-ingredient-contributes). The comparison says nothing about which tool
is better in general.

**Is this a physics-informed neural network (PINN)?**
In the broad sense, yes: physics is built into the architecture (panel-method baseline and features, exact symmetry,
drag prior). In the narrow sense of a PDE-residual loss, no. That option is implemented, but it does not help here
(see [Method](method.md) and the ablation). Of the built-in ingredients only the panel method measurably improves
accuracy; the symmetry and the drag prior make the model consistent rather than more accurate.

**Is the comparison fair to NeuralFoil?**
We tried to make it as fair as we could, and to make every choice checkable.

* Both models get the same exact coordinates and conditions and are scored on the same converged points.
* The XFOIL settings match NeuralFoil's data pipeline: 279 panels, \(N_\mathrm{crit} = 9\), 100 iterations.
* All eight NeuralFoil sizes are reported.
* The comparison is repeated under five definitions of the XFOIL ground truth, including NeuralFoil's own
  warm-started protocol on its own Kulfan geometry.

The remaining asymmetry is intrinsic: we chose the family and the domain. See the
[threats-to-validity table](benchmark.md#threats-to-validity-and-how-each-is-addressed).

**Why a cold start for every XFOIL point?**
So that a label does not depend on which other points were solved, or in what order. This makes data sets
order-independent, parallelisable and extensible (the Sobol prefixes of the data-efficiency curve rely on it). The
cost is that near stall a cold start can land on a different solution branch than a warm-started sweep. The
ground-truth study measures how much that matters.

**Why 279 panels?**
It is AeroSandbox's default re-panelling, which NeuralFoil's training data used. Evaluating against XFOIL at 160
panels (XFOIL's default) would make NeuralFoil look about 10 % worse, which would be unfair to it.

**Can I use it for NACA 5-digit / 6-series / my own airfoil?**
Not with the shipped model. The pipeline is generic in principle: replace the geometry, panel nodes and features, and
retrain. The package does not do this for you. For arbitrary shapes, use NeuralFoil.

**Can I get gradients for optimisation?**
Yes. Everything from the geometry parameters to the outputs is a PyTorch graph; see
[Predictions → Gradients](guide/predictions.md#gradients). The shipped weights are smooth, but keep the
[support flag](guide/predictions.md#knowing-when-not-to-trust-a-prediction) in mind: an optimiser will happily exploit
unsupported regions.

**How fast is it?**
See [Cost per prediction](results.md#cost-per-prediction). It is far faster than XFOIL. Compared with NeuralFoil it
depends on the batch shape: NeuralNACA needs one small panel solve per distinct airfoil, so many angles or Reynolds
numbers on the same section are very cheap.

**The numbers I get differ in the last digits.**
On another platform, retraining changes the weights slightly; see
[Reproducing the results](reproducibility.md#level-1-retrain). Predictions from the *shipped* weights should agree to
about 10⁻⁶. If they do not, please open an issue with the output of `neuralnaca info` and
`neuralnaca reproduce --level 0 --json report.json`.

**Why is `CD` sometimes enormous?**
You are almost certainly outside the training support (deep stall of a thin section at low Reynolds number, where
XFOIL itself does not converge). Check `supported`; an `UnsupportedWarning` should have been raised. See
[Limitations](limitations.md#where-xfoil-does-not-converge-the-model-is-unsupported-and-can-be-absurd).
