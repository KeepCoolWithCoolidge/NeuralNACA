# Running XFOIL

`neuralnaca.xfoil` is a robust batch driver for XFOIL. It is used to build the data sets, but it is also useful on
its own.

```python
import numpy as np
from neuralnaca.xfoil import detect_xfoil, run_xfoil

if __name__ == "__main__":
    cfg = detect_xfoil()                                     # python module if available, else an executable
    pts = np.array([[0.02, 0.4, 0.12, 1e6, a] for a in range(-4, 13)])   # [m, p, t, Re, alpha]
    res = run_xfoil(pts, cfg, workers=4)                     # (n, 3) [CL, CD, CM]; NaN rows = not converged
```

!!! warning "Use a `__main__` guard"
    Points are solved in worker processes started with the `spawn` method, which re-imports your main module. Code
    that calls `run_xfoil` must be protected by `if __name__ == "__main__":` (Jupyter notebooks are fine). If the
    workers cannot start, `run_xfoil` raises an error that says so rather than returning all-NaN results.
    `workers=0` runs in-process (no crash isolation).

## Configuration

`XfoilConfig` fixes everything that influences the answer. The defaults define the ground truth of all published
results.

| field | default | notes |
|---|---|---|
| `backend` | `"module"` | `"module"` (xfoil-python, in-process per worker) or `"binary"` (one process per point) |
| `executable` | `None` | path of the XFOIL program (`binary` backend) |
| `panels` | 279 | nodes after XFOIL's `PPAR` re-panelling. 279 is AeroSandbox's default, used for NeuralFoil's data |
| `n_crit` | 9.0 | e^N transition criterion, free transition |
| `max_iter` | 100 | viscous Newton iterations |
| `timeout` | 30 s | per point (`binary` backend) |
| `protocol` | `"cold"` | `"cold"` or `"nf-sweep"` (below) |
| `geometry` | `"naca"` | `"naca"` (exact coordinates) or `"kulfan"` (NeuralFoil's internal fit) |

## Solution protocols

XFOIL's viscous solution is iterative. Near stall, and in some low-Reynolds-number cases, the result can depend on
where the iteration starts, so the protocol is part of the definition of the ground truth.

* **`cold`** (default): every point starts from a fresh boundary-layer initialisation. A point's answer does not depend
  on which other points were run, so data sets can be generated in any order, in parallel, and extended later.
* **`nf-sweep`**: replicates how NeuralFoil's training data were generated. AeroSandbox's `XFoil.alpha(..., start_at=0)`
  runs a sweep of 7 angles spaced 5 degrees in one XFOIL session, split at zero: positive angles upward from the
  smallest positive one, the others downward after a re-initialisation, each warm-started from the previous
  solution, with 200 iterations. For a single target angle, `neuralnaca.xfoil.sweep_chain` gives the angles
  visited before it.

The [benchmark](../benchmark.md#sensitivity-to-the-ground-truth) quantifies how much the choice matters.

## Geometry representations

* **`naca`**: the exact coordinates from `neuralnaca.naca4_coordinates` (100 points per side, cosine spacing), which
  XFOIL re-panels.
* **`kulfan`**: the geometry NeuralFoil actually analyses. NeuralFoil normalises the coordinates and fits 8+8 Kulfan
  (CST) weights; the fit is evaluated at `alpha + rotation_angle` and `Re / scale`, and the moment is transferred back
  to the original quarter chord, exactly as `neuralfoil.get_aero_from_airfoil` does. Requires `aerosandbox`.

## Robustness

* Worker processes that die inside XFOIL (segmentation fault, floating-point trap) are contained. The chunk is split
  in halves and retried; a single point is retried alone in a private worker, and only a point that kills its own
  isolated worker is reported as not converged. The test suite simulates such crashes.
* Non-converged points, exceptions and non-physical outputs (`CD <= 0`) are all returned as NaN rows.
* The python module leaks a copy of the Fortran library for every new `XFoil()` object, so each worker keeps one
  instance (with a boundary-layer reset before every point) and recycles it every 400 points.

## Data sets

`neuralnaca.data.generate_dataset` samples a design, runs XFOIL and stores an `XfoilDataset` (`.npz` with `points`,
`results` and JSON provenance: design, XFOIL configuration and build, time, platform). A cache file is only reused if it
holds exactly the requested points and XFOIL configuration; otherwise it is regenerated. See
[File formats](../reference/formats.md).
