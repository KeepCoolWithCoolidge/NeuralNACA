"""Regenerate the XFOIL-derived fixtures in tests/data/ (needs an XFOIL 6.99 executable).

    python scripts/make_test_fixtures.py --xfoil /path/to/xfoil

Fixtures
--------
xfoil_inviscid.json          XFOIL *inviscid* CL / CM for a few NACA sections (same coordinates as the package),
                             used to validate the package's own panel method without needing XFOIL at test time.
polar_sample.txt             A genuine XFOIL polar save file (for the parser test).
xfoil_viscous_reference.json 24 benign (attached-flow) points of the shipped test set with their shipped XFOIL
                             results; the XFOIL integration test checks that *your* XFOIL reproduces them.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from neuralnaca.data import XfoilDataset
from neuralnaca.geometry import naca4_coordinates, parse_naca
from neuralnaca.xfoil import parse_polar_text

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "tests" / "data"
SECTIONS = ["0012", "2412", "4412", "6409", "2415", "0006", "3318"]
ALPHAS = [-4.0, 0.0, 4.0, 8.0]


def xfoil_inviscid(exe: str, code: str, panels: int) -> np.ndarray:
    m, p, t = parse_naca(code)
    co = naca4_coordinates(m, p, t)
    with tempfile.TemporaryDirectory() as d:
        foil, pol = os.path.join(d, "foil.dat"), os.path.join(d, "pol.txt")
        Path(foil).write_text("foil\n" + "".join(f"{x:.7f} {y:.7f}\n" for x, y in co))
        cmds = ["PLOP", "G F", "", f"LOAD {foil}", "PPAR", f"N {panels}", "", "", "OPER", "PACC", pol, ""]
        cmds += [f"ALFA {a}" for a in ALPHAS] + ["PACC", "", "QUIT", ""]
        subprocess.run([exe], input="\n".join(cmds), text=True, capture_output=True, timeout=60, cwd=d, check=False)
        return parse_polar_text(Path(pol).read_text())


def polar_sample(exe: str) -> str:
    co = naca4_coordinates(0.02, 0.4, 0.12)
    with tempfile.TemporaryDirectory() as d:
        foil, pol = os.path.join(d, "foil.dat"), os.path.join(d, "pol.txt")
        Path(foil).write_text("foil\n" + "".join(f"{x:.7f} {y:.7f}\n" for x, y in co))
        cmds = ["PLOP", "G F", "", f"LOAD {foil}", "PPAR", "N 279", "", "", "OPER", "VISC 1e6", "ITER 100",
                "PACC", pol, "", "ALFA 0", "ALFA 2", "ALFA 4", "PACC", "", "QUIT", ""]
        subprocess.run([exe], input="\n".join(cmds), text=True, capture_output=True, timeout=60, cwd=d, check=False)
        return Path(pol).read_text()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xfoil", required=True, help="XFOIL 6.99 executable")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    inv = {"description": "XFOIL 6.99 inviscid CL/CM (about x/c = 0.25) for neuralnaca.geometry.naca4_coordinates",
           "alphas": ALPHAS, "sections": {}}
    for code in SECTIONS:
        inv["sections"][code] = {}
        for panels in (160, 279):
            r = xfoil_inviscid(a.xfoil, code, panels)
            inv["sections"][code][str(panels)] = {"alpha": r[:, 0].tolist(), "CL": r[:, 1].tolist(),
                                                  "CM": r[:, 4].tolist()}
    (OUT / "xfoil_inviscid.json").write_text(json.dumps(inv, indent=1) + "\n")

    (OUT / "polar_sample.txt").write_text(polar_sample(a.xfoil))

    ds = XfoilDataset.load(HERE.parent / "data" / "xfoil" / "test_3000_s0_p279_nc9.npz")
    ok = ds.converged & (ds.points[:, 4] > -3) & (ds.points[:, 4] < 7) & (ds.points[:, 3] > 3e5)
    idx = np.where(ok)[0][:24]
    ref = {"description": "benign points of data/xfoil/test_3000_s0_p279_nc9.npz (cold start, 279 panels, n_crit 9)",
           "points": ds.points[idx].tolist(), "results": ds.results[idx].tolist()}
    (OUT / "xfoil_viscous_reference.json").write_text(json.dumps(ref, indent=1) + "\n")
    print("fixtures written to", OUT)


if __name__ == "__main__":
    main()
