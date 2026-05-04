"""
Run the 12 CRD variants on the next 5 CEC2017 functions (indices 10..14).
Settings: pop=30, max_iter=300, n_runs=30, dim=30.

Each function runs in its own process so the 5 functions execute in parallel.
Output layout:
    experiments/CRD_CEC2017_D30/<CEC17_F*>/CSV Data, Convergence Curves, Box Plots, Excel Files
"""

from __future__ import annotations

import multiprocessing
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEURILAB_ROOT = Path("c:/Users/ROG SRTIX/Desktop/pkg/heurilab")
for p in (PROJECT_ROOT, HEURILAB_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from heurilab import run_experiment
from heurilab.core.benchmarks import BenchmarkSuite
from heurilab.core.cec2017 import CEC2017_FUNCTIONS

from algorithms import CRD_VARIANTS


# ── Settings ────────────────────────────────────────────────────────────────
POP = 30
ITERS = 300
N_RUNS = 30
DIM = 30
START_IDX = 10        # picks up where the first-10 script left off
N_FUNC = 5            # next 5 functions: indices 10..14
OUTPUT_ROOT = str(PROJECT_ROOT / "experiments/CRD_CEC2017_D30")

# All 12 CRD variants — keep RD off to focus on CRD only
ALGORITHMS = [(name, cls) for name, cls in CRD_VARIANTS.items()]


def run_single(index: int, dim: int = DIM) -> None:
    name, func, lb, ub, _native_dim = CEC2017_FUNCTIONS[index]
    safe_name = name.replace(" ", "_").replace(":", "")

    suite = BenchmarkSuite(f"CEC2017_D{dim}")
    suite.add(safe_name, func, lb, ub, dim=dim)

    print(f"[Process] Starting {safe_name} D={dim}")
    run_experiment(
        algorithms=ALGORITHMS,
        benchmark_suites=[suite],
        output_dir=f"{OUTPUT_ROOT}/{safe_name}",
        pop_size=POP,
        max_iter=ITERS,
        dim=dim,
        n_runs=N_RUNS,
        run_engineering=False,
    )
    print(f"[Process] Finished {safe_name} D={dim}")


if __name__ == "__main__":
    processes = []
    for idx in range(START_IDX, START_IDX + N_FUNC):
        p = multiprocessing.Process(target=run_single, args=(idx, DIM))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    print(f"CRD CEC2017 D={DIM} (idx {START_IDX}..{START_IDX + N_FUNC - 1}) done!")
