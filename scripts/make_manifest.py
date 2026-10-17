"""Write benchmarks/MANIFEST.json: SHA-256 of every shipped data, model, result and weights file.

    python scripts/make_manifest.py          # after (re)running experiments
    python -c "from neuralnaca.data import verify_manifest; print(verify_manifest('benchmarks/MANIFEST.json'))"
"""

from __future__ import annotations

import json
from pathlib import Path

from neuralnaca.data import sha256

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks"
PATTERNS = ["data/xfoil/*.npz", "benchmarks/models/*.npz", "benchmarks/results/*.json", "benchmarks/results/*.npz",
            "src/neuralnaca/weights/*.npz", "tests/data/*"]


def main() -> None:
    files = {}
    for pattern in PATTERNS:
        for p in sorted(ROOT.glob(pattern)):
            files[Path("..", p.relative_to(ROOT)).as_posix()] = sha256(p)
    out = BENCH / "MANIFEST.json"
    out.write_text(json.dumps({"description": "SHA-256 of every shipped file; paths are relative to this file",
                               "files": files}, indent=1) + "\n")
    print(f"wrote {out} ({len(files)} files)")


if __name__ == "__main__":
    main()
