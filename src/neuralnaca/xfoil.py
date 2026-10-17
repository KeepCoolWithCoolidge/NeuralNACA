"""Running XFOIL: the "teacher" that generates training data and the ground truth for evaluation.

Two interchangeable backends are supported:

``"module"``
    The ``xfoil`` Python extension (https://github.com/daniel-de-vries/xfoil-python), in-process. About four times
    faster than the executable. One persistent instance per worker process (every new ``XFoil()`` object leaks a
    copy of the Fortran library), recycled every 400 points.
``"binary"``
    An XFOIL 6.99 executable, one process per point (crash- and hang-proof). The polar save file is parsed.

Two solution protocols:

``"cold"`` (default, used for all training and test data)
    Every point is solved independently from a fresh boundary-layer initialisation. The answer for a point does not
    depend on which other points were run.
``"nf-sweep"``
    Replicates the protocol of NeuralFoil's training data (``asb.XFoil.alpha`` with ``start_at=0``): points are
    reached by an angle-of-attack sweep in ``sweep_step`` increments that starts next to 0 deg on the same side of
    zero and is warm-started from the previous converged point; ``max_iter`` is 200. Used only in the
    ground-truth sensitivity study.

and two geometry representations:

``"naca"``
    The exact NACA 4-digit coordinates (:func:`neuralnaca.geometry.naca4_coordinates`).
``"kulfan"``
    The 8+8-weight Kulfan (CST) fit that NeuralFoil makes of those coordinates internally, including NeuralFoil's
    normalisation (rotation / scaling) and its moment-reference correction - i.e. the geometry NeuralFoil actually
    analyses. Needs ``aerosandbox``. Used only in the ground-truth sensitivity study.
``"naca-normalized"``
    Control case: the *exact* NACA coordinates passed through the same normalisation and frame transformation as
    ``"kulfan"`` but without the Kulfan fit. Separates the effect of the shape fit from that of the frame handling.

Workers that die inside XFOIL (segmentation faults, floating-point traps) are contained by :func:`run_xfoil`: failed
chunks are split and retried, and a point that kills an isolated worker is reported as not converged.
"""

from __future__ import annotations

import concurrent.futures as cf
import logging
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace

import numpy as np

from .geometry import naca4_coordinates

__all__ = [
    "PROTOCOLS",
    "XfoilConfig",
    "XfoilNotFoundError",
    "detect_xfoil",
    "kulfan_equivalent",
    "normalized_equivalent",
    "parse_polar_text",
    "run_xfoil",
    "solve_point",
    "sweep_chain",
    "xfoil_version",
]

log = logging.getLogger(__name__)

PROTOCOLS = ("cold", "nf-sweep")
GEOMETRIES = ("naca", "kulfan", "naca-normalized")
NAN3 = (math.nan, math.nan, math.nan)

INSTALL_HELP = """No working XFOIL found. Install ONE of:
  (a) the Python module (fastest):  scripts/install_xfoil_python.sh
      = pip install of https://github.com/daniel-de-vries/xfoil-python built WITHOUT its floating-point-trap
        compiler flags (needs cmake + gfortran). See docs/installation.md.
  (b) an XFOIL 6.99 executable on PATH (or pass executable=/path/to/xfoil):  scripts/build_xfoil_binary.sh
      Note: some distribution builds (e.g. Debian/Ubuntu 'apt install xfoil') abort with a floating-point
      exception in the viscous solver; rebuild without '-ffpe-trap'. On a headless machine wrap the run with
      'xvfb-run -a' if your build needs X11."""


class XfoilNotFoundError(RuntimeError):
    """No usable XFOIL backend is available."""


@dataclass(frozen=True)
class XfoilConfig:
    """How XFOIL is run. The defaults define the ground truth of every reported result.

    Attributes
    ----------
    backend
        ``"module"`` or ``"binary"``.
    executable
        Path of the XFOIL executable (``binary`` backend).
    panels
        Number of panel nodes after XFOIL's ``PPAR`` re-panelling (279 = AeroSandbox's default, which NeuralFoil's
        data pipeline uses).
    n_crit
        e^N transition criterion (free transition on both surfaces).
    max_iter
        Viscous Newton iterations per point.
    timeout
        Seconds per point (``binary`` backend).
    protocol
        ``"cold"`` or ``"nf-sweep"`` (see module docstring).
    sweep_step
        Angle increment of the ``"nf-sweep"`` protocol (NeuralFoil: 7 angles spaced 5 deg).
    geometry
        ``"naca"`` or ``"kulfan"`` (see module docstring).
    """

    backend: str = "module"
    executable: str | None = None
    panels: int = 279
    n_crit: float = 9.0
    max_iter: int = 100
    timeout: float = 30.0
    protocol: str = "cold"
    sweep_step: float = 5.0
    geometry: str = "naca"

    def __post_init__(self):
        if self.backend not in ("module", "binary"):
            raise ValueError(f"unknown XFOIL backend {self.backend!r}")
        if self.protocol not in PROTOCOLS:
            raise ValueError(f"unknown protocol {self.protocol!r}; choose from {PROTOCOLS}")
        if self.geometry not in GEOMETRIES:
            raise ValueError(f"unknown geometry {self.geometry!r}; choose from {GEOMETRIES}")
        if self.backend == "binary" and not self.executable:
            raise ValueError("the binary backend needs `executable`")

    def tag(self) -> str:
        """Short identifier used in data file names; ``p279_nc9`` for the default ground truth."""
        s = f"p{self.panels}_nc{self.n_crit:g}"
        if self.protocol != "cold" or self.geometry != "naca" or self.max_iter != 100:
            s += f"_{self.protocol}_{self.geometry}_it{self.max_iter}"
        return s

    def neuralfoil_protocol(self, geometry: str | None = None) -> XfoilConfig:
        """The same backend with NeuralFoil's data-generation protocol (warm-started sweeps, 200 iterations)."""
        return replace(self, protocol="nf-sweep", max_iter=200, geometry=geometry or self.geometry)

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------------------------------------ case preparation
def sweep_chain(alpha: float, step: float = 5.0) -> list[float]:
    """Angles visited by AeroSandbox's ``alpha(..., start_at=0)`` sweep before (and including) ``alpha``.

    Positive targets are approached from the smallest positive member of the sweep upwards, non-positive targets
    from the member closest to zero downwards (after a boundary-layer re-initialisation), with ``step`` spacing.

    >>> sweep_chain(12.3)
    [2.3, 7.3, 12.3]
    >>> sweep_chain(-7.0)
    [-2.0, -7.0]
    >>> sweep_chain(4.0)
    [4.0]
    """
    if alpha > 0:
        k = math.ceil(alpha / step - 1e-12) - 1
        chain = [alpha - step * j for j in range(k, 0, -1)] + [alpha]
    else:
        k = math.floor(-alpha / step + 1e-12)
        chain = [alpha + step * j for j in range(k, 0, -1)] + [alpha]
    return [round(a, 6) for a in chain]


def kulfan_equivalent(m: float, p: float, t: float) -> tuple[np.ndarray, dict]:
    """The geometry NeuralFoil analyses for a NACA section, and the frame transformation it applies.

    Mirrors ``neuralfoil.get_aero_from_airfoil``: normalise the coordinates (leading edge at the origin, chord 1,
    trailing edge on the x axis), fit 8+8 Kulfan weights, and analyse the fitted shape at
    ``alpha + rotation_angle`` and ``Re / scale``; the moment is then transferred back to the original quarter chord.

    Returns
    -------
    coords : numpy.ndarray (N, 2)
        Selig-ordered coordinates of the fitted Kulfan airfoil.
    frame : dict
        ``delta_alpha`` [deg], ``scale``, ``x_qc``, ``y_qc`` such that
        ``CM_original = CM + (-CL * x_qc + CD * y_qc)``.
    """
    import aerosandbox as asb

    af = asb.Airfoil(coordinates=naca4_coordinates(m, p, t))
    norm = af.normalize(return_dict=True)
    kaf = norm["airfoil"].to_kulfan_airfoil(n_weights_per_side=8, normalize_coordinates=False)
    da, scale = float(norm["rotation_angle"]), float(norm["scale_factor"])
    x_qc = -float(norm["x_translation"]) + 0.25 * (1.0 / scale * math.cos(math.radians(da))) - 0.25
    y_qc = -float(norm["y_translation"]) + 0.25 * (1.0 / scale * math.sin(math.radians(-da)))
    return np.asarray(kaf.coordinates, dtype=float), {"delta_alpha": da, "scale": scale, "x_qc": x_qc, "y_qc": y_qc}


def normalized_equivalent(m: float, p: float, t: float) -> tuple[np.ndarray, dict]:
    """Exact NACA coordinates in NeuralFoil's normalised frame (no Kulfan fit); same return values as
    :func:`kulfan_equivalent`."""
    import aerosandbox as asb

    norm = asb.Airfoil(coordinates=naca4_coordinates(m, p, t)).normalize(return_dict=True)
    da, scale = float(norm["rotation_angle"]), float(norm["scale_factor"])
    x_qc = -float(norm["x_translation"]) + 0.25 * (1.0 / scale * math.cos(math.radians(da))) - 0.25
    y_qc = -float(norm["y_translation"]) + 0.25 * (1.0 / scale * math.sin(math.radians(-da)))
    return np.asarray(norm["airfoil"].coordinates, dtype=float), {"delta_alpha": da, "scale": scale, "x_qc": x_qc,
                                                                   "y_qc": y_qc}


def _prepare(pt: Sequence[float], cfg: XfoilConfig):
    """-> coordinates, list of angles to run (target last), Reynolds number, post-processing function."""
    m, p, t, re, alpha = (float(v) for v in pt)
    if cfg.geometry in ("kulfan", "naca-normalized"):
        co, fr = (kulfan_equivalent if cfg.geometry == "kulfan" else normalized_equivalent)(m, p, t)
        alpha_x, re_x = alpha + fr["delta_alpha"], re / fr["scale"]

        def post(cl, cd, cm):
            return cl, cd, cm + (-cl * fr["x_qc"] + cd * fr["y_qc"])
    else:
        co, alpha_x, re_x = naca4_coordinates(m, p, t), alpha, re

        def post(cl, cd, cm):
            return cl, cd, cm
    if cfg.protocol == "nf-sweep":
        # the sweep is defined in the user's frame (as in NeuralFoil's data generation), then shifted
        chain = [a + (alpha_x - alpha) for a in sweep_chain(alpha, cfg.sweep_step)]
    else:
        chain = [alpha_x]
    return co, chain, re_x, post


# ------------------------------------------------------------------------------------------------ point solvers
_XF: dict = {"obj": None, "n": 0}


def _module_solver(pt: Sequence[float], cfg: XfoilConfig) -> tuple[float, float, float]:
    from xfoil import XFoil
    from xfoil.model import Airfoil

    if _XF["obj"] is None or _XF["n"] >= 400:
        _XF["obj"] = XFoil()
        _XF["obj"].print = False
        _XF["n"] = 0
    _XF["n"] += 1
    xf = _XF["obj"]
    co, chain, re, post = _prepare(pt, cfg)
    xf.airfoil = Airfoil(co[:, 0], co[:, 1])
    xf.repanel(n_nodes=cfg.panels)
    xf.Re, xf.M, xf.n_crit, xf.max_iter = re, 0.0, cfg.n_crit, cfg.max_iter
    xf.reset_bls()                                              # cold start: independent of previously solved points
    cl = cd = cm = math.nan
    for a in chain:                                             # warm-started continuation for "nf-sweep"
        cl, cd, cm, _ = xf.a(a)
    return post(cl, cd, cm)


def parse_polar_text(text: str) -> np.ndarray:
    """Parse an XFOIL polar save file (``PACC``) into rows ``[alpha, CL, CD, CDp, CM, ...]``.

    Header, separator and parameter lines are skipped: a data row is any line whose first five fields are numbers.
    """
    rows = []
    for line in text.splitlines():
        s = line.split()
        if len(s) < 5:
            continue
        try:
            rows.append([float(v) for v in s[:5]])
        except ValueError:
            continue
    return np.array(rows, dtype=float).reshape(-1, 5)


def _binary_solver(pt: Sequence[float], cfg: XfoilConfig) -> tuple[float, float, float]:
    co, chain, re, post = _prepare(pt, cfg)
    with tempfile.TemporaryDirectory() as d:
        foil, pol = os.path.join(d, "foil.dat"), os.path.join(d, "pol.txt")
        with open(foil, "w") as f:
            f.write("foil\n" + "".join(f"{x:.7f} {y:.7f}\n" for x, y in co))
        cmds = ["PLOP", "G F", "", f"LOAD {foil}", "foil", "PPAR", f"N {cfg.panels}", "", "",
                "OPER", f"VISC {re:.6e}", "MACH 0", "VPAR", f"N {cfg.n_crit}", "", f"ITER {cfg.max_iter}",
                "PACC", pol, ""]
        cmds += [f"ALFA {a:.4f}" for a in chain]
        cmds += ["PACC", "", "QUIT", ""]
        try:
            subprocess.run([str(cfg.executable)], input="\n".join(cmds), text=True, capture_output=True,
                           timeout=cfg.timeout, cwd=d, check=False)
        except (subprocess.TimeoutExpired, OSError):
            return NAN3
        if not os.path.exists(pol):
            return NAN3
        with open(pol) as fh:
            rows = parse_polar_text(fh.read())
    target = round(chain[-1], 4)
    hit = rows[np.abs(rows[:, 0] - target) < 2e-3] if len(rows) else rows
    if not len(hit):
        return NAN3                                             # target angle did not converge
    _, cl, cd, _, cm = hit[-1]
    return post(cl, cd, cm)


def solve_point(pt: Sequence[float], cfg: XfoilConfig) -> tuple[float, float, float]:
    """Solve one condition ``[m, p, t, Re, alpha]`` with the configured backend; NaNs if not converged."""
    fn = _module_solver if cfg.backend == "module" else _binary_solver
    try:
        r = fn(pt, cfg)
    except Exception:                                           # a failed XFOIL point is "not converged"
        return NAN3
    if not (np.isfinite(r[0]) and np.isfinite(r[2]) and r[1] > 0):
        return NAN3
    return float(r[0]), float(r[1]), float(r[2])


Solver = Callable[[Sequence[float], XfoilConfig], tuple[float, float, float]]


def _run_chunk(args) -> list[tuple[float, float, float]]:
    pts, cfg, solver = args
    fn = solver or solve_point
    out = []
    for pt in pts:
        try:
            r = fn(tuple(pt), cfg)
        except Exception:
            r = NAN3
        if not (np.isfinite(r[0]) and r[1] > 0):
            r = NAN3
        out.append(tuple(r))
    return out


def _probe() -> int:
    return os.getpid()


def _check_workers_can_start(ctx) -> None:
    """Fail loudly (instead of silently recording every point as 'not converged') if workers cannot start.

    The usual cause is a script without an ``if __name__ == "__main__":`` guard, or code fed to Python on stdin:
    spawned workers re-import the main module and die.
    """
    try:
        with cf.ProcessPoolExecutor(max_workers=1, mp_context=ctx) as ex:
            ex.submit(_probe).result(timeout=300)
    except Exception as e:
        raise RuntimeError(
            "XFOIL worker processes could not be started. Run your code from a file protected by\n"
            "    if __name__ == '__main__':\n"
            "(required by Python's 'spawn' multiprocessing), or pass workers=0 to run XFOIL in-process without\n"
            "crash isolation.") from e


def run_xfoil(points: np.ndarray, cfg: XfoilConfig, workers: int | None = None, label: str = "xfoil",
              chunk: int = 32, solver: Solver | None = None) -> np.ndarray:
    """Solve many conditions in parallel.

    Parameters
    ----------
    points
        ``(n, 5)`` array ``[m, p, t, Re, alpha]``.
    cfg
        XFOIL settings.
    workers
        Worker processes (default: all CPUs). ``0`` runs in the calling process (no crash isolation: a fatal XFOIL
        error then terminates the caller), which also works where child processes cannot be spawned.
    label
        Progress-log label.
    chunk
        Points per task.
    solver
        Optional replacement for :func:`solve_point` (a picklable module-level function); used by the tests.

    Returns
    -------
    numpy.ndarray, shape (n, 3)
        ``[CL, CD, CM]``; rows of NaN where XFOIL did not converge (or crashed).

    Notes
    -----
    Worker crashes are contained: a chunk whose worker died is split in halves and retried; a single point is
    retried alone in a private one-worker pool (it may only have been collateral damage of another point), and a
    point that kills its own isolated worker is reported as NaN.
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    if workers == 0:                                            # in-process, no crash isolation
        return np.array(_run_chunk((points, cfg, solver)), dtype=float).reshape(n, 3)
    workers = workers or max(1, os.cpu_count() or 1)
    res = np.full((n, 3), np.nan)
    pending = [(a, min(a + chunk, n)) for a in range(0, n, chunk)]
    ctx = mp.get_context("spawn")
    _check_workers_can_start(ctx)
    done, t0, last = 0, time.time(), 0.0
    while pending:
        finished = set()
        try:
            with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
                futs = {ex.submit(_run_chunk, (points[a:b], cfg, solver)): (a, b) for a, b in pending}
                for f in cf.as_completed(futs):
                    a, b = futs[f]
                    try:
                        res[a:b] = np.array(f.result())
                    except Exception:                           # worker died (e.g. SIGFPE / SIGSEGV inside XFOIL)
                        continue
                    finished.add((a, b))
                    done += b - a
                    if time.time() - last > 15 or done == n:
                        last = time.time()
                        rate = done / max(time.time() - t0, 1e-9)
                        log.info("[%s] XFOIL %d/%d  (%.1f pts/s, ETA %.1f min)", label, done, n, rate,
                                 (n - done) / max(rate, 1e-9) / 60)
        except Exception:
            pass                                                # pool-level failure: unfinished chunks are retried
        failed = [c for c in pending if c not in finished]
        pending = []
        for a, b in failed:
            if b - a > 1:                                       # unknown culprit: halve the chunk and retry
                mid = (a + b) // 2
                pending += [(a, mid), (mid, b)]
                continue
            try:                                                # isolated retry of a single point
                with cf.ProcessPoolExecutor(max_workers=1, mp_context=ctx) as ex:
                    res[a:b] = np.array(ex.submit(_run_chunk, (points[a:b], cfg, solver)).result())
            except Exception:
                log.warning("[%s] point %d crashed XFOIL in isolation; recorded as not converged", label, a)
            done += 1
    return res


# ------------------------------------------------------------------------------------------------ discovery
def detect_xfoil(backend: str = "auto", executable: str | None = None, **kwargs) -> XfoilConfig:
    """Find a usable XFOIL backend.

    Parameters
    ----------
    backend
        ``"auto"`` (prefer the module, fall back to an executable), ``"module"`` or ``"binary"``.
    executable
        Explicit path of the XFOIL executable; otherwise ``$NEURALNACA_XFOIL``, then ``xfoil`` on ``PATH``.
    **kwargs
        Further :class:`XfoilConfig` fields (``panels``, ``protocol``, ...).

    Raises
    ------
    XfoilNotFoundError
        With installation instructions.
    """
    if backend not in ("auto", "module", "binary"):
        raise ValueError(f"unknown backend {backend!r}")
    if backend in ("auto", "module"):
        try:
            import xfoil  # noqa: F401

            return XfoilConfig(backend="module", **kwargs)
        except ImportError:
            if backend == "module":
                raise XfoilNotFoundError("the `xfoil` Python module is not importable.\n" + INSTALL_HELP) from None
    path = executable or os.environ.get("NEURALNACA_XFOIL") or shutil.which("xfoil") or shutil.which("xfoil.exe")
    if path:
        return XfoilConfig(backend="binary", executable=path, **kwargs)
    raise XfoilNotFoundError(INSTALL_HELP)


def xfoil_version(cfg: XfoilConfig) -> str:
    """Human-readable description of the XFOIL build behind ``cfg`` (for provenance records)."""
    if cfg.backend == "module":
        try:
            from importlib.metadata import version

            return f"xfoil-python {version('xfoil')} (XFOIL 6.99 Fortran core)"
        except Exception:
            return "xfoil-python (version unknown)"
    try:
        out = subprocess.run([str(cfg.executable)], input="QUIT\n", text=True, capture_output=True, timeout=10,
                             check=False).stdout
        for line in out.splitlines():
            if "Version" in line:
                return f"{line.strip()} ({cfg.executable})"
    except Exception:
        pass
    return f"unknown ({cfg.executable})"
