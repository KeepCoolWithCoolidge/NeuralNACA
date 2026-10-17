# Training your own model

The shipped model was trained on 4,096 XFOIL runs. The same pipeline trains a model on any data set you generate,
for instance with more runs, another domain or another transition criterion.

## Command line

```bash
# 1. sample a space-filling design and run XFOIL on it (needs XFOIL)
neuralnaca generate --n 8192 --kind sobol --seed 7 --out my_train.npz
neuralnaca generate --n 2000 --kind random --seed 1007 --out my_test.npz

# 2. train (5-member ensemble, 18,000 steps: ~5-7 min on 2 CPU cores)
neuralnaca train --data my_train.npz --out my_model.npz

# 3. evaluate against XFOIL (and NeuralFoil)
neuralnaca evaluate --weights my_model.npz --data my_test.npz --neuralfoil xxxlarge

# 4. use it
neuralnaca predict 2412 --re 1e6 --weights my_model.npz
```

`train` also writes `my_model.support.npz` (the [training-support](predictions.md#knowing-when-not-to-trust-a-prediction)
model), which `NeuralNACA.load("my_model.npz")` picks up automatically.

!!! tip "Designs"
    Use `--kind sobol` with a power of two for training sets: scrambled Sobol points fill the domain evenly, and every
    prefix of length 2^k is itself balanced (the data-efficiency study relies on this). Use `--kind random` for
    test sets, so that the test error is an unbiased estimate of the domain-averaged error, and use a different seed
    than for training.

## Python

```python
from neuralnaca.data import generate_dataset, pack_dataset
from neuralnaca.xfoil import detect_xfoil
from neuralnaca.training import TrainConfig, train_ensemble
from neuralnaca.model import save_model
from neuralnaca.support import SupportModel

if __name__ == "__main__":                       # required: XFOIL runs in spawned worker processes
    cfg = detect_xfoil()
    ds = generate_dataset(4096, "sobol", seed=0, cfg=cfg, path="train.npz")

    data = pack_dataset(ds)                       # converged points only (the others stay in the file)
    net = train_ensemble(data, TrainConfig(seed=0))
    save_model(net, "model.npz", {"note": "my model"})
    SupportModel.from_dataset(ds).save("model.support.npz")
```

## Hyper-parameters

`TrainConfig` holds the recipe; its defaults are exactly what produced every published number (a test fails if they
change).

| field | default | meaning |
|---|---|---|
| `ensemble` | 5 | independent networks trained together |
| `steps` | 18000 | optimiser steps (one-cycle schedule) |
| `batch_size` | 512 | minibatch per member (each member draws its own) |
| `lr` | 3e-3 | peak learning rate (Adam) |
| `huber_beta` | 0.1 | smooth-L1 transition on the scaled outputs |
| `ema` | 0.9995 | weight averaging over the second half of training |
| `width`, `depth` | 128, 4 | hidden units / layers (SiLU) |
| `variant` | `"full"` | `"plain"`, `"no_panel"`, `"no_symmetry"`, `"no_drag_prior"` remove one ingredient (ablations) |
| `residuals` | False | add the autograd physics residuals (needs collocation points) |

Training is deterministic for a given seed on a given machine and PyTorch build (the test suite checks this).
Other machines may differ in the last bits, which the optimiser amplifies; [Reproducing the results](../reproducibility.md)
quantifies the resulting spread.

## Changing the domain

`neuralnaca.domain.Domain` describes the box. Pass a custom domain to `sample_points` / `generate_dataset` to sample
it, and to `NeuralNACA(..., domain=...)` so that predictions are checked against it. The input features are scaled
for the default box; moderate changes are fine, but large changes (for example `t` up to 0.30) deserve a look at
`neuralnaca.model.make_inputs`.
