"""Predict a few conditions and a polar with the shipped model."""

import numpy as np

import neuralnaca

if __name__ == "__main__":
    aero = neuralnaca.get_aero("NACA 2412", alpha=4.0, Re=1e6)
    print(f"NACA 2412, alpha = 4 deg, Re = 1e6:  CL = {float(aero['CL']):.4f}  CD = {float(aero['CD']):.5f}  "
          f"CM = {float(aero['CM']):.4f}  (supported: {bool(aero['supported'])})")

    model = neuralnaca.NeuralNACA.pretrained()
    polar = model.polar("4412", Re=3e5, alpha=np.arange(-4.0, 14.1, 2.0))
    print(f"\n{model}\nNACA 4412 at Re = 3e5")
    for a, cl, cd, cm, s in zip(polar["alpha"], polar["CL"], polar["CD"], polar["CM"], polar["CL_std"]):
        print(f"  alpha {a:5.1f}   CL {cl:7.4f} (+-{s:.4f})   CD {cd:8.5f}   L/D {cl / cd:6.1f}   CM {cm:7.4f}")
