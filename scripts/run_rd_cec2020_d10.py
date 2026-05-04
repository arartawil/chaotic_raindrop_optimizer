"""
Run the ORIGINAL Raindrop Optimizer (RD) on all 10 CEC2020 functions at D = 10.
No chaotic variants, no competitors — just the base RD algorithm.

Settings: pop=30, max_iter=300, n_runs=30.

Each function runs in its own process so the 10 functions execute in parallel.
Output layout:
    experiments/RD_CEC2020_D10/<CEC20_F*>/CSV Data, Convergence Curves, Box Plots, Excel Files
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
from heurilab.core.cec2020 import CEC2020_FUNCTIONS

from algorithms import RD


# ── Settings ────────────────────────────────────────────────────────────────
POP = 30
ITERS = 300
N_RUNS = 30
DIM = 10                                     # CEC2020 official dims: 5, 10, 15, 20
N_FUNC = len(CEC2020_FUNCTIONS)              # 10
OUTPUT_ROOT = str(PROJECT_ROOT / f"experiments/RD_CEC2020_D{DIM}")

# Just the original RD
ALGORITHMS = [("RD", RD)]


def run_single(index: int, dim: int = DIM) -> None:
    name, func, lb, ub, _native_dim = CEC2020_FUNCTIONS[index]
    safe_name = name.replace(" ", "_").replace(":", "")

    suite = BenchmarkSuite(f"CEC2020_D{dim}")
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
    for idx in range(N_FUNC):
        p = multiprocessing.Process(target=run_single, args=(idx, DIM))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    print(f"RD CEC2020 D={DIM} (F1..F{N_FUNC}) done!")
