"""Compare NeuralNACA and NeuralFoil on one section (needs `pip install neuralfoil`)."""

import numpy as np

from neuralnaca import NeuralNACA, naca4_coordinates, parse_naca

if __name__ == "__main__":
    import neuralfoil as nf

    code, re = "2412", 1e6
    alpha = np.arange(-4.0, 12.1, 2.0)
    ours = NeuralNACA.pretrained().predict(code, alpha=alpha, Re=re)
    theirs = nf.get_aero_from_coordinates(naca4_coordinates(*parse_naca(code)), alpha=alpha, Re=re, n_crit=9,
                                          model_size="xxxlarge")
    print(f"NACA {code}, Re = {re:.0e}        NeuralNACA                NeuralFoil xxxlarge")
    for i, a in enumerate(alpha):
        print(f"  alpha {a:5.1f}   CL {ours['CL'][i]:7.4f} CD {ours['CD'][i]:8.5f}   "
              f"CL {float(theirs['CL'][i]):7.4f} CD {float(theirs['CD'][i]):8.5f}")
    print("\nFor agreement with XFOIL itself, see docs/results.md or run `neuralnaca predict ... --compare-xfoil`.")
