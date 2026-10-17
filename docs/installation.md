# Installation

NeuralNACA needs Python ≥ 3.10. What else you install depends on what you want to do:

| goal | install | needs XFOIL? |
|---|---|---|
| predict polars with the shipped model | `pip install neuralnaca` (or from the repository, below) | no |
| compare against NeuralFoil, regenerate figures | `pip install "neuralnaca[neuralfoil,plot]"` | no |
| verify / retrain the published results (levels 0-1) | repository checkout + `pip install -e ".[neuralfoil,plot,test]"` | no |
| generate new XFOIL data, reproduce level 2 | the above + XFOIL (below) | **yes** |

```bash
git clone https://github.com/KeepCoolWithCoolidge/NeuralNACA.git
cd NeuralNACA
python -m pip install -e ".[dev]"      # everything: neuralfoil, matplotlib, test and docs tools
neuralnaca info                        # versions, shipped model fingerprint, XFOIL availability
```

The runtime dependencies are only NumPy, SciPy and PyTorch (CPU is enough; the model is small).

## XFOIL

XFOIL is the "teacher" that produced every training and test label. It is **not** needed to make predictions or
to verify the published numbers from the shipped data, only to generate new data. Two interchangeable backends are
supported; you need one of them.

### Option A: the `xfoil` Python module (recommended, about 4x faster)

```bash
bash scripts/install_xfoil_python.sh
neuralnaca xfoil-check --backend module
```

The script clones [xfoil-python](https://github.com/daniel-de-vries/xfoil-python) at the exact commit used for the
benchmark (`5c8c4ff`), removes its floating-point-trap compiler flags and installs it with pip. It needs `git`,
`cmake`, `gfortran` and a C compiler (Debian/Ubuntu: `sudo apt-get install git cmake gfortran build-essential`;
macOS: `brew install cmake gcc`).

!!! note "Why patch the compiler flags?"
    Upstream compiles XFOIL with `-ffpe-trap=invalid,zero -finit-real=inf -fbounds-check`. With traps enabled, a
    floating-point exception inside the viscous solver kills the whole Python process instead of merely failing to
    converge. The benchmark data were generated without traps (`-O2 -fdefault-real-8`). We verified that a fresh
    build made with the script reproduces the shipped XFOIL results **bit for bit**.

### Option B: an XFOIL 6.99 executable

```bash
bash scripts/build_xfoil_binary.sh ~/opt/xfoil          # Debian/Ubuntu: builds from the distribution's source package
export NEURALNACA_XFOIL=~/opt/xfoil/xfoil
neuralnaca xfoil-check --backend binary
```

The executable is found through `executable=...`, then `$NEURALNACA_XFOIL`, then `xfoil` on the `PATH`. Every point
runs in its own XFOIL process, which is slower but immune to crashes and hangs.

!!! warning "Distribution packages can be broken"
    The binary shipped by Debian/Ubuntu (`apt install xfoil`) aborts with a floating-point exception on most viscous
    points. `neuralnaca xfoil-check` detects this (it prints `FAIL` and NaNs). Rebuild it with the script above. On
    other systems, build the [official sources](https://web.mit.edu/drela/Public/web/xfoil/) after removing
    `-ffpe-trap=...`, `-finit-real=inf` and `-fbounds-check` from the Makefiles (and adding
    `-fallow-argument-mismatch` for gfortran ≥ 10). The script reproduces the benchmark's binary byte for byte on
    Ubuntu.

### Checking an XFOIL installation

`neuralnaca xfoil-check` solves three reference conditions and compares them with the build used for the benchmark
(tolerances 0.005 in C<sub>L</sub>, 0.0002 in C<sub>D</sub>, 0.003 in C<sub>M</sub>). The test suite goes further:
`pytest -m xfoil` checks that your XFOIL reproduces 24 points of the shipped test set and exercises both solution
protocols.

## Supported platforms

The package is pure Python plus PyTorch and runs wherever PyTorch runs (Linux, macOS, Windows). The published results
were produced on Linux x86-64; see [Reproducing the results](reproducibility.md) for what "reproducible" means across
platforms.

## Building the documentation

```bash
pip install -e ".[docs]"
mkdocs serve          # or: mkdocs build --strict
```
