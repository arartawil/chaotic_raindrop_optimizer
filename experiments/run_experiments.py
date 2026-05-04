"""
Per-function multiprocessing runner for the CRD experiments.

Mirrors the heurilab template:

    def run_single(index, dim):
        functions = get_cec2022_functions(dim=dim)
        name, func, lb, ub, bias = functions[index]
        suite = BenchmarkSuite(f"CEC2022_D{dim}")
        suite.add(safe_name, func, lb, ub, dim=dim)
        run_experiment(algorithms=[...], benchmark_suites=[suite], ...)

Each (suite, dim, function-index) runs in its own process so the 12 CEC2022
functions (or the 29 CEC2017 ones) are processed in parallel.

Output layout::

    experiments/CRD_CEC2022_D10/F1/
    experiments/CRD_CEC2022_D10/F2/
    ...
    experiments/CRD_CEC2017_D30/CEC17_F1_BentCigar/
    ...

Each subfolder contains heurilab's CSV / Convergence Curves / Box Plots /
Excel Files.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path
from typing import List, Tuple, Type

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEURILAB_ROOT = Path("c:/Users/ROG SRTIX/Desktop/pkg/heurilab")
PHAGE_ROOT = Path("c:/Users/ROG SRTIX/Desktop/pkg/Phage Lifecycle")
for p in (PROJECT_ROOT, HEURILAB_ROOT, PHAGE_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


# ── Defaults from CRD_Claude_Code_Prompt.md ─────────────────────────────────
POP = 50
N_RUNS = 30
ITERS_CEC2017 = 500
ITERS_CEC2022 = 1000
ALG_TAG = "CRD"  # used in the output dir name


# ────────────────────────────────────────────────────────────────────────────
def _get_algorithms(no_competitors: bool, crd_subset: List[str] | None):
    from experiments.algorithm_roster import build_roster

    return build_roster(
        include_competitors=not no_competitors,
        include_crd=True,
        crd_subset=crd_subset,
    )


def _get_cec2022_functions(dim: int):
    from cec2022 import get_cec2022_functions

    return get_cec2022_functions(dim=dim)


def _get_cec2017_functions(dim: int):
    """Return [(name, func, lb, ub, dim)] for CEC2017 at a given dim."""
    from heurilab.core.cec2017 import CEC2017_FUNCTIONS

    return [(name, func, lb, ub, dim) for name, func, lb, ub, _ in CEC2017_FUNCTIONS]


# ── Per-function process bodies ─────────────────────────────────────────────
def run_single_cec2022(index: int, dim: int, *,
                       output_root: str, pop: int, iters: int, n_runs: int,
                       no_competitors: bool, crd_subset: List[str] | None,
                       run_engineering: bool) -> None:
    """One process: full algorithm roster on a single CEC2022 function."""
    from heurilab import run_experiment
    from heurilab.core.benchmarks import BenchmarkSuite

    functions = _get_cec2022_functions(dim=dim)
    name, func, lb, ub, _bias = functions[index]
    safe_name = name.split(":", 1)[0].strip().replace(" ", "_")

    suite = BenchmarkSuite(f"CEC2022_D{dim}")
    suite.add(safe_name, func, lb, ub, dim=dim)

    algorithms = _get_algorithms(no_competitors, crd_subset)

    out_dir = f"{output_root}/{ALG_TAG}_CEC2022_D{dim}/{safe_name}"
    print(f"[Process] Starting {safe_name} Dim={dim}  algorithms={len(algorithms)}")
    run_experiment(
        algorithms=algorithms,
        benchmark_suites=[suite],
        output_dir=out_dir,
        pop_size=pop,
        max_iter=iters,
        dim=dim,
        n_runs=n_runs,
        run_engineering=run_engineering,
        engineering_n_runs=n_runs,
        engineering_max_iter=iters,
    )
    print(f"[Process] Finished {safe_name} Dim={dim}")


def run_single_cec2017(index: int, dim: int, *,
                       output_root: str, pop: int, iters: int, n_runs: int,
                       no_competitors: bool, crd_subset: List[str] | None,
                       run_engineering: bool) -> None:
    """One process: full algorithm roster on a single CEC2017 function."""
    from heurilab import run_experiment
    from heurilab.core.benchmarks import BenchmarkSuite

    functions = _get_cec2017_functions(dim=dim)
    name, func, lb, ub, fdim = functions[index]
    safe_name = name.replace(" ", "_").replace(":", "")

    suite = BenchmarkSuite(f"CEC2017_D{dim}")
    suite.add(safe_name, func, lb, ub, dim=dim)

    algorithms = _get_algorithms(no_competitors, crd_subset)

    out_dir = f"{output_root}/{ALG_TAG}_CEC2017_D{dim}/{safe_name}"
    print(f"[Process] Starting {safe_name} Dim={dim}  algorithms={len(algorithms)}")
    run_experiment(
        algorithms=algorithms,
        benchmark_suites=[suite],
        output_dir=out_dir,
        pop_size=pop,
        max_iter=iters,
        dim=dim,
        n_runs=n_runs,
        run_engineering=run_engineering,
        engineering_n_runs=n_runs,
        engineering_max_iter=iters,
    )
    print(f"[Process] Finished {safe_name} Dim={dim}")


# ── Main ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "experiments_output"))
    p.add_argument("--runs", type=int, default=N_RUNS)
    p.add_argument("--pop", type=int, default=POP)
    p.add_argument("--iters-cec2017", type=int, default=ITERS_CEC2017)
    p.add_argument("--iters-cec2022", type=int, default=ITERS_CEC2022)
    p.add_argument("--cec2017-dims", nargs="+", type=int, default=[30])
    p.add_argument("--cec2022-dims", nargs="+", type=int, default=[10, 20])
    p.add_argument("--cec2017-only", action="store_true")
    p.add_argument("--cec2022-only", action="store_true")
    p.add_argument("--no-competitors", action="store_true")
    p.add_argument("--no-engineering", action="store_true")
    p.add_argument("--crd-subset", nargs="+", default=None)
    p.add_argument("--max-procs", type=int, default=None,
                   help="cap concurrent function processes; default = all functions at once")
    p.add_argument("--quick", action="store_true",
                   help="3 funcs / 80 iters / smaller budget for smoke runs")
    p.add_argument("--cec2022-only-funcs", nargs="+", type=int, default=None,
                   help="restrict CEC2022 to these function indices (0-based)")
    p.add_argument("--cec2017-only-funcs", nargs="+", type=int, default=None,
                   help="restrict CEC2017 to these function indices (0-based)")
    return p.parse_args()


def _launch(targets: List[mp.Process], max_procs: int | None) -> None:
    if not targets:
        return
    if max_procs is None or max_procs >= len(targets):
        for p in targets:
            p.start()
        for p in targets:
            p.join()
        return
    # bounded pool
    pending = list(targets)
    running: List[mp.Process] = []
    while pending or running:
        while pending and len(running) < max_procs:
            p = pending.pop(0)
            p.start()
            running.append(p)
        # poll
        finished = [p for p in running if not p.is_alive()]
        for p in finished:
            p.join()
            running.remove(p)
        if running and not finished:
            running[0].join(0.1)


def main():
    args = parse_args()

    iters_2017 = 80 if args.quick else args.iters_cec2017
    iters_2022 = 80 if args.quick else args.iters_cec2022
    n_runs = 2 if args.quick else args.runs
    pop = 15 if args.quick else args.pop

    common_kwargs = dict(
        output_root=str(args.out_dir),
        pop=pop,
        n_runs=n_runs,
        no_competitors=args.no_competitors,
        crd_subset=args.crd_subset,
        run_engineering=not args.no_engineering,
    )

    procs: List[mp.Process] = []

    # CEC2022 — 12 functions per dim
    if not args.cec2017_only:
        from cec2022 import get_cec2022_functions  # noqa: F401  (early import sanity check)

        for dim in args.cec2022_dims:
            n_funcs = 12
            indices = args.cec2022_only_funcs or list(range(n_funcs))
            if args.quick:
                indices = indices[:3]
            for idx in indices:
                p = mp.Process(
                    target=run_single_cec2022,
                    args=(idx, dim),
                    kwargs={**common_kwargs, "iters": iters_2022},
                )
                procs.append(p)

    # CEC2017 — 29 functions per dim
    if not args.cec2022_only:
        for dim in args.cec2017_dims:
            n_funcs = 29
            indices = args.cec2017_only_funcs or list(range(n_funcs))
            if args.quick:
                indices = indices[:3]
            for idx in indices:
                p = mp.Process(
                    target=run_single_cec2017,
                    args=(idx, dim),
                    kwargs={**common_kwargs, "iters": iters_2017},
                )
                procs.append(p)

    print(f"[run] launching {len(procs)} processes (max_procs={args.max_procs or 'all'})")
    _launch(procs, args.max_procs)
    print(f"[run] all done — outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
