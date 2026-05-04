"""Algorithm roster: RD + 12 CRD variants + 10 competitor metaheuristics."""

from __future__ import annotations

from typing import List, Tuple, Type

from heurilab.algorithms import (
    PSO, GWO, WOA, SCA, MFO, HHO, SSA, AOA, SAO, DE,
)

from algorithms import RD, CRD_VARIANTS


# 1) Proposed RD + 12 chaotic variants
PROPOSED_ALGORITHMS: List[Tuple[str, Type]] = [("RD", RD)]
PROPOSED_ALGORITHMS += [(name, cls) for name, cls in CRD_VARIANTS.items()]

# 2) Competitor metaheuristics (Task 9)
COMPETITOR_ALGORITHMS: List[Tuple[str, Type]] = [
    ("PSO", PSO),
    ("GWO", GWO),
    ("WOA", WOA),
    ("SCA", SCA),
    ("MFO", MFO),
    ("HHO", HHO),
    ("SSA", SSA),
    ("AOA", AOA),
    ("SAO", SAO),
    ("DE", DE),
]


def build_roster(
    include_competitors: bool = True,
    include_crd: bool = True,
    crd_subset: List[str] | None = None,
) -> List[Tuple[str, Type]]:
    """Return a (name, class) list ready for `heurilab.run_experiment`."""
    roster: List[Tuple[str, Type]] = [("RD", RD)]
    if include_crd:
        for name, cls in CRD_VARIANTS.items():
            if crd_subset is None or name in crd_subset:
                roster.append((name, cls))
    if include_competitors:
        roster.extend(COMPETITOR_ALGORITHMS)
    return roster
