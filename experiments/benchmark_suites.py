"""Build heurilab `BenchmarkSuite` objects for CEC2017 and CEC2022."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import List

from heurilab.core.benchmarks import BenchmarkSuite
from heurilab.core.cec2017 import CEC2017_FUNCTIONS as _HEURILAB_CEC2017_FUNCTIONS

# CEC2022 lives in the Phage Lifecycle pkg next door — import it lazily.
_PHAGE_LIFECYCLE_PATH = Path("c:/Users/ROG SRTIX/Desktop/pkg/Phage Lifecycle")


def _load_cec2022_module():
    if not _PHAGE_LIFECYCLE_PATH.exists():
        raise RuntimeError(
            f"Cannot find Phage Lifecycle pkg at {_PHAGE_LIFECYCLE_PATH}. "
            "CEC2022 benchmark unavailable."
        )
    if str(_PHAGE_LIFECYCLE_PATH) not in sys.path:
        sys.path.insert(0, str(_PHAGE_LIFECYCLE_PATH))
    import cec2022  # noqa: E402

    return cec2022


def cec2017_suite(dim: int = 30) -> BenchmarkSuite:
    """Build a heurilab CEC2017 suite at the requested dimension.

    heurilab's bundled list is fixed at dim=30; we re-use the function pointers
    but override the per-benchmark dim so the suite can run at 50 or 100 too.
    """
    suite = BenchmarkSuite(category=f"CEC2017_D{dim}")
    for name, func, lb, ub, _native_dim in _HEURILAB_CEC2017_FUNCTIONS:
        suite.add(name, func, lb, ub, dim=dim)
    return suite


def cec2022_suite(dim: int = 10) -> BenchmarkSuite:
    """Build a heurilab BenchmarkSuite from Phage Lifecycle's cec2022 module."""
    cec = _load_cec2022_module()
    funcs = cec.get_cec2022_functions(dim=dim)
    suite = BenchmarkSuite(category=f"CEC2022_D{dim}")
    for name, func, lb, ub, _bias in funcs:
        # `name` looks like "F1: <description>"; keep the prefix only
        clean_name = name.split(":", 1)[0].strip()
        suite.add(clean_name, func, lb, ub, dim=dim)
    return suite


def all_suites(
    cec2017_dim: int = 30,
    cec2022_dims: List[int] = (10, 20),
) -> List[BenchmarkSuite]:
    suites: List[BenchmarkSuite] = []
    suites.append(cec2017_suite(dim=cec2017_dim))
    for d in cec2022_dims:
        try:
            suites.append(cec2022_suite(dim=d))
        except Exception as exc:
            print(f"[suites] skipping CEC2022 D={d}: {exc}")
    return suites
