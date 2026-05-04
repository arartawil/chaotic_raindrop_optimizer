"""
Run the ORIGINAL Raindrop Optimizer (RD) on the LAST CEC2017 functions
(indices 20..28). heurilab's CEC2017 has 29 functions total, so after the
prior batches (0..9 and 10..19) only 9 remain.

Settings: pop=30, max_iter=300, n_runs=30, dim=30.

Each function runs in its own process so all 9 functions execute in parallel.
Output layout:
    experiments/RD_CEC2017_D30/<CEC17_F*>/CSV Data, Convergence Curves, Box Plots, Excel Files
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

from algorithms import RD


# ── Settings ────────────────────────────────────────────────────────────────
POP = 30
ITERS = 300
N_RUNS = 30
DIM = 30
START_IDX = 20
END_IDX = len(CEC2017_FUNCTIONS)            # 29 → runs indices 20..28
OUTPUT_ROOT = str(PROJECT_ROOT / f"experiments/RD_CEC2017_D{DIM}")

# Just the original RD
ALGORITHMS = [("RD", RD)]


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
    for idx in range(START_IDX, END_IDX):
        p = multiprocessing.Process(target=run_single, args=(idx, DIM))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    print(f"RD CEC2017 D={DIM} (idx {START_IDX}..{END_IDX - 1}) done!")
