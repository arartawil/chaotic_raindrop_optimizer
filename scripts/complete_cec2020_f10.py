"""
Finish the partial F10 (Composition3) results in
    experiments/CRD_CEC2020_D10/CEC20_F10_Composition3/

Tasks:
  • CRD-Piecewise   : already has 23/30 runs → run 7 more (runs 24..30)
  • CRD-Iterative   : 0/30 runs → run all 30
  • CRD-LogisticTent: 0/30 runs → run all 30

Appends the new rows to the existing raw_runs.csv and convergence.csv,
then rewrites the per-algorithm summary rows in results.csv so they reflect
all 30 runs.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEURILAB_ROOT = Path("c:/Users/ROG SRTIX/Desktop/pkg/heurilab")
for p in (PROJECT_ROOT, HEURILAB_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from heurilab.core.cec2020 import CEC2020_FUNCTIONS
from heurilab.exporters.csv_export import append_raw_run, _pad_or_trim

from algorithms import CRD_VARIANTS


# ── Settings (must match the original run_crd_cec2020_d10.py) ───────────────
POP = 30
ITERS = 300
DIM = 10
F_INDEX = 9                      # F10 in the CEC2020 list (0-based: 9)
OUTPUT_ROOT = PROJECT_ROOT / f"experiments/CRD_CEC2020_D{DIM}"

# What to fix
TARGETS = {
    "CRD-Piecewise":    {"missing_runs": list(range(23, 30))},  # 7 runs
    "CRD-Iterative":    {"missing_runs": list(range(0, 30))},   # 30 runs
    "CRD-LogisticTent": {"missing_runs": list(range(0, 30))},   # 30 runs
}


def _load_function(index: int):
    name, func, lb, ub, _native_dim = CEC2020_FUNCTIONS[index]
    safe_name = name.replace(" ", "_").replace(":", "")
    return safe_name, func, lb, ub


def _run_one(algo_class, func, lb, ub, dim, pop, iters):
    algo = algo_class(
        pop_size=pop, dim=dim, lb=lb, ub=ub, max_iter=iters, obj_func=func,
    )
    t0 = time.time()
    _, best_fit, conv = algo.optimize()
    return float(best_fit), time.time() - t0, list(conv)


def _rewrite_results_csv(results_path: Path, raw_runs_path: Path,
                         benchmark: str, algorithms_to_refresh: List[str]):
    """Replace each algo's summary row in results.csv from the raw_runs data."""
    df = pd.read_csv(raw_runs_path)
    df = df[df["Benchmark"] == benchmark]

    # Read existing results
    if results_path.exists():
        res = pd.read_csv(results_path)
    else:
        res = pd.DataFrame(columns=["Benchmark", "Algorithm", "Mean", "Std", "Best", "Worst", "Median"])

    for algo in algorithms_to_refresh:
        sub = df[df["Algorithm"] == algo]["BestFitness"].values
        if len(sub) == 0:
            continue
        new_row = {
            "Benchmark": benchmark,
            "Algorithm": algo,
            "Mean":   f"{np.mean(sub):.6e}",
            "Std":    f"{np.std(sub):.6e}",
            "Best":   f"{np.min(sub):.6e}",
            "Worst":  f"{np.max(sub):.6e}",
            "Median": f"{np.median(sub):.6e}",
        }
        # remove existing row(s) for (benchmark, algo) and append the new one
        mask = ~((res["Benchmark"] == benchmark) & (res["Algorithm"] == algo))
        res = pd.concat([res[mask], pd.DataFrame([new_row])], ignore_index=True)

    res.to_csv(results_path, index=False)


def _rewrite_convergence_csv(conv_path: Path, raw_runs_path: Path,
                             benchmark: str, algorithms_to_refresh: List[str],
                             max_iter: int):
    """Replace each algo's mean-convergence row using all runs in raw_runs.csv."""
    df = pd.read_csv(raw_runs_path)
    df = df[df["Benchmark"] == benchmark]
    conv_cols = [f"Conv_{i}" for i in range(max_iter + 1)]

    if conv_path.exists():
        cv = pd.read_csv(conv_path)
    else:
        iter_cols = [f"Iter_{i}" for i in range(max_iter + 1)]
        cv = pd.DataFrame(columns=["Benchmark", "Algorithm"] + iter_cols)

    iter_cols = [f"Iter_{i}" for i in range(max_iter + 1)]
    for algo in algorithms_to_refresh:
        sub = df[df["Algorithm"] == algo][conv_cols].values
        if len(sub) == 0:
            continue
        # pad to max_iter+1 if needed
        padded = np.array([_pad_or_trim(list(row), max_iter + 1) for row in sub], dtype=float)
        mean_conv = np.nanmean(padded, axis=0)
        new_row = {"Benchmark": benchmark, "Algorithm": algo}
        for col, v in zip(iter_cols, mean_conv):
            new_row[col] = f"{v:.6e}"
        mask = ~((cv["Benchmark"] == benchmark) & (cv["Algorithm"] == algo))
        cv = pd.concat([cv[mask], pd.DataFrame([new_row])], ignore_index=True)

    cv.to_csv(conv_path, index=False)


def main():
    safe_name, func, lb, ub = _load_function(F_INDEX)
    out_dir = OUTPUT_ROOT / safe_name
    csv_dir = out_dir / "CSV Data"
    if not csv_dir.exists():
        raise SystemExit(f"Missing folder: {csv_dir}")

    raw_path  = csv_dir / "raw_runs.csv"
    res_path  = csv_dir / "results.csv"
    conv_path = csv_dir / "convergence.csv"

    print(f"[fix] target = {safe_name} (D={DIM}, pop={POP}, iters={ITERS})")

    refreshed = []

    # 1) Append all missing runs to raw_runs.csv
    for algo_name, info in TARGETS.items():
        algo_class = CRD_VARIANTS[algo_name]
        for run_idx in info["missing_runs"]:
            best_fit, elapsed, conv = _run_one(algo_class, func, lb, ub, DIM, POP, ITERS)
            append_raw_run(
                str(raw_path),
                benchmark_name=safe_name,
                algo_name=algo_name,
                run_idx=run_idx,         # 0-based; helper writes run_idx+1
                best_fitness=best_fit,
                elapsed=elapsed,
                convergence=conv,
                max_iter=ITERS,
            )
            print(f"  [{algo_name}] run {run_idx + 1}/30  best={best_fit:.6e}  t={elapsed:.2f}s")
        refreshed.append(algo_name)

    # 2) Rewrite results.csv summary rows for the touched algorithms
    _rewrite_results_csv(res_path, raw_path, safe_name, refreshed)
    print(f"[fix] results.csv updated for {refreshed}")

    # 3) Rewrite convergence.csv mean rows for the touched algorithms
    _rewrite_convergence_csv(conv_path, raw_path, safe_name, refreshed, ITERS)
    print(f"[fix] convergence.csv updated for {refreshed}")

    print("[fix] done.")


if __name__ == "__main__":
    main()
