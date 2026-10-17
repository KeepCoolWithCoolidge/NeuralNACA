#!/usr/bin/env bash
# Build and install the `xfoil` Python module exactly as it was used to generate the benchmark data.
#
#   bash scripts/install_xfoil_python.sh            # installs into the active Python environment
#
# Requirements: git, cmake, a Fortran compiler (gfortran), a C compiler, Python >= 3.10 with pip.
#   Debian/Ubuntu: sudo apt-get install git cmake gfortran build-essential
#   macOS (Homebrew): brew install cmake gcc
#
# Why a patch? The upstream CMakeLists.txt compiles XFOIL with floating-point traps
# (-ffpe-trap=invalid,zero -finit-real=inf -fbounds-check). With traps, a floating-point exception inside the
# viscous solver kills the whole Python process instead of just failing to converge. The benchmark data were
# generated WITHOUT traps (-O2 -fdefault-real-8); numerics of converged points are unaffected.
set -euo pipefail
COMMIT=5c8c4ff62d1b9db03f3b8711f38941cd411ae892     # daniel-de-vries/xfoil-python master, 2019-12-11
PYTHON=${PYTHON:-python3}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

git clone --quiet https://github.com/daniel-de-vries/xfoil-python.git "$WORK/xfoil-python"
cd "$WORK/xfoil-python"
git checkout --quiet "$COMMIT"

"$PYTHON" - <<'PY'
import re, pathlib
p = pathlib.Path("CMakeLists.txt")
s = p.read_text()
new = re.sub(r'set\(CMAKE_Fortran_FLAGS "\$\{CMAKE_Fortran_FLAGS\} \\\n-O \\\n-fbounds-check \\\n-finit-real=inf \\\n'
             r'-ffpe-trap=invalid,zero \\\n-fdefault-real-8"\)',
             'set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -O2 -fdefault-real-8")', s)
if new == s:
    raise SystemExit("could not find the expected Fortran flags in CMakeLists.txt - upstream changed?")
p.write_text(new)
print("patched CMakeLists.txt: floating-point traps removed")
PY

"$PYTHON" -m pip install ${PIP_FLAGS:-} .
cd /    # import the installed module, not the source tree
"$PYTHON" -c "import xfoil; print('xfoil module installed:', xfoil.__file__)"
echo "Now run:  neuralnaca xfoil-check --backend module"
