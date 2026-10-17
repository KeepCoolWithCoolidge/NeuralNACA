"""NeuralNACA - a physics-embedded neural surrogate for NACA 4-digit airfoil polars.

Quick start
-----------
>>> import neuralnaca
>>> aero = neuralnaca.get_aero("NACA 2412", alpha=4.0, Re=1e6)
>>> float(aero["CL"]), float(aero["CD"]), float(aero["CM"])  # doctest: +SKIP
(0.72, 0.0071, -0.057)

The model reproduces XFOIL (``n_crit = 9``, Mach 0, free transition) inside the validated domain
:data:`DEFAULT_DOMAIN`; see the documentation for the benchmark against NeuralFoil and its caveats.
"""

from ._version import __version__
from .api import DEFAULT_WEIGHTS, NeuralNACA, get_aero
from .domain import DEFAULT_DOMAIN, Domain, DomainWarning, UnsupportedWarning, sample_points
from .geometry import format_naca, naca4_coordinates, parse_naca

__all__ = [
    "DEFAULT_DOMAIN",
    "DEFAULT_WEIGHTS",
    "Domain",
    "DomainWarning",
    "NeuralNACA",
    "UnsupportedWarning",
    "__version__",
    "format_naca",
    "get_aero",
    "naca4_coordinates",
    "parse_naca",
    "sample_points",
]
