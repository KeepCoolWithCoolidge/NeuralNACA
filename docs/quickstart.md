# Quick start

## In Python

```python
import numpy as np
import neuralnaca

# one condition: scalar in, scalars out
aero = neuralnaca.get_aero("NACA 2412", alpha=4.0, Re=1e6)
print(float(aero["CL"]), float(aero["CD"]), float(aero["CM"]))

# a polar: arrays broadcast like NumPy
alpha = np.arange(-4.0, 12.1, 1.0)
polar = neuralnaca.get_aero("4412", alpha=alpha, Re=3e5)
print(polar["CL"].shape)                      # (17,)

# non-integer sections: give m, p, t directly (fractions of chord)
aero = neuralnaca.get_aero(m=0.025, p=0.45, t=0.105, alpha=2.0, Re=2e6)
```

Every call returns a dictionary of NumPy arrays with the broadcast shape of the inputs:

| key | meaning |
|---|---|
| `CL`, `CD`, `CM` | lift, drag and quarter-chord pitching-moment coefficients (ensemble mean; `CD = exp(mean ln CD)`) |
| `CL_std`, `lnCD_std`, `CD_std`, `CM_std` | spread of the 5 ensemble members (a heuristic error indicator, not a calibrated interval) |
| `in_domain` | the condition lies inside the [validated domain](guide/predictions.md#the-validated-domain) |
| `xfoil_convergence` | estimated probability that XFOIL itself converges here (from the training data) |
| `supported` | in the domain **and** backed by training data; see [Knowing when not to trust a prediction](guide/predictions.md#knowing-when-not-to-trust-a-prediction) |

Conditions outside the domain or without training support raise a `DomainWarning` / `UnsupportedWarning` (both are
`UserWarning`s); pass `on_domain_error="raise"` to turn them into errors or `"ignore"` to silence them.

For repeated use, keep a model object (this is what `get_aero` does internally):

```python
from neuralnaca import NeuralNACA

model = NeuralNACA.pretrained()
print(model)                                   # variant, ensemble size and weight fingerprint
out = model.predict("0012", alpha=[-2.0, 0.0, 2.0], Re=5e5)
pol = model.polar("6409", Re=1e6)              # alpha from -5 to 15 deg in 0.5 deg steps
```

## On the command line

```console
$ neuralnaca predict 2412 --re 1e6 --alpha -4 12 --step 2
NACA 2412  Re = 1e+06  (n_crit = 9, Mach 0, free transition)
  alpha       CL        CD       CM     +-CL  +-CD%
  -4.00  -0.1937   0.00761  -0.0551   0.0007   0.20
  -2.00   0.0255   0.00646  -0.0537   0.0002   0.26
  ...
$ neuralnaca predict 2412 --re 1e6 --alpha 0 4 8 --json        # machine-readable
$ neuralnaca predict 2412 --re 1e6 --compare-xfoil              # side by side with XFOIL (needs XFOIL)
```

## Next steps

* What the model is and why it works: [Method](method.md).
* How it compares with NeuralFoil, and how that comparison was made: [Benchmark protocol](benchmark.md) and
  [Results](results.md).
* Where it should not be trusted: [Limitations](limitations.md).
* Re-deriving every published number yourself: [Reproducing the results](reproducibility.md).
