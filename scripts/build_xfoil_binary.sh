#!/usr/bin/env bash
# Build an XFOIL 6.99 executable from the Debian/Ubuntu source package without floating-point traps - the build
# used for the benchmark's cross-checks. (The distribution's own binary package aborts with SIGFPE in the viscous
# solver on many points; `neuralnaca xfoil-check` detects that.)
#
#   bash scripts/build_xfoil_binary.sh [install-dir]      # default install-dir: ./xfoil-6.99
#
# Requirements (Debian/Ubuntu): deb-src entries enabled, and
#   sudo apt-get install dpkg-dev gfortran make libx11-dev
# Other platforms: build the official sources from https://web.mit.edu/drela/Public/web/xfoil/ the same way -
# remove "-ffpe-trap=..." / "-finit-real=inf" / "-fbounds-check" from the Makefiles and add
# "-fallow-argument-mismatch" for gfortran >= 10.
set -euo pipefail
DEST=$(realpath -m "${1:-./xfoil-6.99}")
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"
apt-get source --quiet xfoil >/dev/null
SRC=$(find . -maxdepth 1 -type d -name 'xfoil-6.99*' | head -1)
[ -n "$SRC" ] || { echo "apt-get source xfoil did not produce a source tree (enable deb-src?)"; exit 1; }
cd "$SRC"
sed -i 's/^CHK = .*/CHK = -fallow-argument-mismatch/' bin/Makefile
sed -i 's/-fbounds-check -finit-real=inf -ffpe-trap=invalid,zero //' orrs/bin/Makefile_DP
make -s -C orrs/bin -f Makefile_DP osgen
(cd orrs && bin/osgen osmaps_ns.lst >/dev/null)
make -s -C plotlib
make -s -C bin xfoil
mkdir -p "$DEST"
cp bin/xfoil "$DEST/xfoil"
echo "built $DEST/xfoil"
echo "Use it with:  export NEURALNACA_XFOIL=$DEST/xfoil && neuralnaca xfoil-check --backend binary"
