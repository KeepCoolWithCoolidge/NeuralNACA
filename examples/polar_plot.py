"""Plot lift, drag polar and moment of a few sections at one Reynolds number (needs matplotlib)."""

import sys

import matplotlib.pyplot as plt
import numpy as np

from neuralnaca import NeuralNACA

if __name__ == "__main__":
    re = float(sys.argv[1]) if len(sys.argv) > 1 else 5e5
    model = NeuralNACA.pretrained()
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(12, 3.8))
    for code in ("0012", "2412", "4415"):
        p = model.polar(code, Re=re, alpha=np.arange(-5, 15.01, 0.25), on_domain_error="ignore")
        ok = p["supported"]
        a1.plot(p["alpha"][ok], p["CL"][ok], label=f"NACA {code}")
        a2.plot(p["CD"][ok], p["CL"][ok])
        a3.plot(p["alpha"][ok], p["CM"][ok])
    a1.set(xlabel="alpha [deg]", ylabel="CL")
    a2.set(xlabel="CD", ylabel="CL")
    a3.set(xlabel="alpha [deg]", ylabel="CM (c/4)")
    a1.legend()
    fig.suptitle(f"NeuralNACA polars, Re = {re:.0e} (supported predictions only)")
    fig.tight_layout()
    fig.savefig("polars.png", dpi=120)
    print("wrote polars.png")
