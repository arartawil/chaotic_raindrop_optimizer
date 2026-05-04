"""
Merge RD + 12 CRD results into one combined output per suite.

Reads:
    experiments/RD_<SUITE>_D<dim>/<func>/CSV Data/raw_runs.csv
    experiments/CRD_<SUITE>_D<dim>/<func>/CSV Data/raw_runs.csv

Writes (per suite):
    experiments/RD_CRD_MERGED_<SUITE>_D<dim>/
        <func>/CSV Data/{raw_runs,results,convergence}.csv
        <func>/Convergence Curves/<func>.png
        <func>/Box Plots/<func>.png
        Excel Files/{Results,Wilcoxon,Friedman}.xlsx        (suite-level)

The Excel files cover ALL functions in the suite at once (one sheet per
suite category, plus a Ranking sheet) — exactly the layout heurilab's
``run_experiment`` would emit if RD and the 12 CRDs had been run together.

Algorithm order in the merged outputs is:
    RD, CRD-Logistic, CRD-Tent, CRD-Sine, CRD-Singer, CRD-Sinusoidal,
    CRD-Chebyshev, CRD-Circle, CRD-Gauss, CRD-Bernoulli, CRD-Piecewise,
    CRD-Iterative, CRD-LogisticTent
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEURILAB_ROOT = Path("c:/Users/ROG SRTIX/Desktop/pkg/heurilab")
for p in (PROJECT_ROOT, HEURILAB_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from heurilab.exporters.excel_export import (
    generate_results_excel, generate_wilcoxon_excel, generate_friedman_excel,
)
from heurilab.exporters.plots import plot_convergence, plot_boxplot

from algorithms import CRD_VARIANTS


EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Algorithm order: RD first (= "proposed"), then the 12 CRD variants
ALGO_ORDER: List[str] = ["RD"] + list(CRD_VARIANTS.keys())

# (suite_tag, dim) — only suites that have BOTH RD and CRD folders are merged
SUITES = [
    ("CEC2017", 30),
    ("CEC2020", 10),
]


def _suite_folder(prefix: str, suite_tag: str, dim: int) -> Path:
    return EXPERIMENTS_DIR / f"{prefix}_{suite_tag}_D{dim}"


def _list_functions(rd_root: Path, crd_root: Path) -> List[str]:
    """Function folders that exist in BOTH RD and CRD roots."""
    rd_funcs = {p.name for p in rd_root.iterdir() if p.is_dir()} if rd_root.exists() else set()
    crd_funcs = {p.name for p in crd_root.iterdir() if p.is_dir()} if crd_root.exists() else set()
    return sorted(rd_funcs & crd_funcs)


def _read_raw_runs(folder: Path) -> pd.DataFrame:
    p = folder / "CSV Data" / "raw_runs.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _write_merged_csvs(merged: pd.DataFrame, out_dir: Path, max_iter: int) -> None:
    """Write raw_runs.csv, results.csv, convergence.csv into out_dir/CSV Data/."""
    csv_dir = out_dir / "CSV Data"
    csv_dir.mkdir(parents=True, exist_ok=True)

    # raw_runs.csv — preserve original columns and order
    merged.to_csv(csv_dir / "raw_runs.csv", index=False)

    # results.csv
    rows = []
    for algo in [a for a in ALGO_ORDER if a in merged["Algorithm"].unique()]:
        sub = merged[merged["Algorithm"] == algo]["BestFitness"].astype(float).values
        if len(sub) == 0:
            continue
        bench = merged[merged["Algorithm"] == algo]["Benchmark"].iloc[0]
        rows.append([
            bench, algo,
            f"{np.mean(sub):.6e}", f"{np.std(sub):.6e}",
            f"{np.min(sub):.6e}", f"{np.max(sub):.6e}",
            f"{np.median(sub):.6e}",
        ])
    with open(csv_dir / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Benchmark", "Algorithm", "Mean", "Std", "Best", "Worst", "Median"])
        w.writerows(rows)

    # convergence.csv (mean curve per algorithm)
    conv_cols = [c for c in merged.columns if c.startswith("Conv_")]
    iter_cols = [f"Iter_{i}" for i in range(len(conv_cols))]
    with open(csv_dir / "convergence.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Benchmark", "Algorithm"] + iter_cols)
        for algo in [a for a in ALGO_ORDER if a in merged["Algorithm"].unique()]:
            rows_a = merged[merged["Algorithm"] == algo]
            if rows_a.empty:
                continue
            arr = rows_a[conv_cols].astype(float).values
            mean_curve = np.nanmean(arr, axis=0)
            bench = rows_a["Benchmark"].iloc[0]
            w.writerow([bench, algo] + [f"{v:.6e}" for v in mean_curve])


def _replot(merged: pd.DataFrame, out_dir: Path, func_name: str) -> None:
    """Re-emit convergence curve and boxplot for the merged set."""
    conv_cols = [c for c in merged.columns if c.startswith("Conv_")]
    max_iter = len(conv_cols) - 1

    # mean convergence per algorithm (preserve ALGO_ORDER)
    mean_conv: Dict[str, list] = {}
    for algo in [a for a in ALGO_ORDER if a in merged["Algorithm"].unique()]:
        arr = merged[merged["Algorithm"] == algo][conv_cols].astype(float).values
        if len(arr) == 0:
            continue
        mean_conv[algo] = np.nanmean(arr, axis=0).tolist()

    # raw fitness lists per algorithm
    raw: Dict[str, List[float]] = {}
    for algo in [a for a in ALGO_ORDER if a in merged["Algorithm"].unique()]:
        raw[algo] = merged[merged["Algorithm"] == algo]["BestFitness"].astype(float).tolist()

    plot_convergence(func_name, list(mean_conv.keys()), mean_conv, str(out_dir), max_iter)
    plot_boxplot(func_name, list(raw.keys()), raw, str(out_dir))


def merge_suite(suite_tag: str, dim: int) -> None:
    rd_root = _suite_folder("RD", suite_tag, dim)
    crd_root = _suite_folder("CRD", suite_tag, dim)
    out_root = EXPERIMENTS_DIR / f"RD_CRD_MERGED_{suite_tag}_D{dim}"

    funcs = _list_functions(rd_root, crd_root)
    if not funcs:
        print(f"[merge] {suite_tag} D={dim}: no overlapping function folders, skip")
        return

    print(f"[merge] {suite_tag} D={dim}: {len(funcs)} functions  ->  {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    # collectors for the suite-level Excel files
    results_data: Dict[str, Dict[str, Dict[str, float]]] = {}
    raw_data: Dict[str, Dict[str, List[float]]] = {}
    suite_funcs: List[str] = []

    for func in funcs:
        rd_df = _read_raw_runs(rd_root / func)
        crd_df = _read_raw_runs(crd_root / func)
        if rd_df.empty and crd_df.empty:
            continue

        # union of columns (raw_runs may differ slightly between RD and CRD if
        # someone changed max_iter — align on the wider set, fill missing)
        merged = pd.concat([rd_df, crd_df], ignore_index=True, sort=False)

        # keep only ALGO_ORDER algorithms, in that order
        merged = merged[merged["Algorithm"].isin(ALGO_ORDER)].copy()
        algo_rank = {a: i for i, a in enumerate(ALGO_ORDER)}
        merged["__order"] = merged["Algorithm"].map(algo_rank)
        merged = merged.sort_values(["__order", "Run"]).drop(columns="__order")

        if merged.empty:
            continue

        out_func = out_root / func
        out_func.mkdir(parents=True, exist_ok=True)

        # write merged CSVs + replot per-function
        conv_cols = [c for c in merged.columns if c.startswith("Conv_")]
        max_iter = len(conv_cols) - 1
        _write_merged_csvs(merged, out_func, max_iter)
        _replot(merged, out_func, func)

        # collect for suite-level Excel
        suite_funcs.append(func)
        results_data[func] = {}
        raw_data[func] = {}
        for algo in [a for a in ALGO_ORDER if a in merged["Algorithm"].unique()]:
            vals = merged[merged["Algorithm"] == algo]["BestFitness"].astype(float).values
            results_data[func][algo] = dict(
                mean=float(np.mean(vals)),
                std=float(np.std(vals)),
                best=float(np.min(vals)),
                worst=float(np.max(vals)),
                median=float(np.median(vals)),
            )
            raw_data[func][algo] = vals.tolist()

    if not suite_funcs:
        print(f"[merge] {suite_tag} D={dim}: nothing to write")
        return

    # Suite-level Excel files (one xls bundle per suite)
    suites_map = {f"{suite_tag}_D{dim}": suite_funcs}
    algo_names_in_data = [a for a in ALGO_ORDER if any(a in r for r in results_data.values())]

    generate_results_excel(results_data, algo_names_in_data, suites_map, str(out_root))
    generate_wilcoxon_excel(raw_data, algo_names_in_data, suites_map, str(out_root))
    generate_friedman_excel(raw_data, algo_names_in_data, suites_map, str(out_root))
    print(f"[merge] {suite_tag} D={dim}: wrote Excel Files/{{Results,Wilcoxon,Friedman}}.xlsx")


def merge_all_suites_into_one_xls() -> None:
    """Combine every suite's per-function CSVs into a SINGLE Excel bundle.

    Produces:
        experiments/RD_CRD_MERGED_ALL/Excel Files/{Results,Wilcoxon,Friedman}.xlsx
    Each workbook holds one sheet per suite (CEC2017_D30, CEC2020_D10),
    plus an "All Functions" sheet, plus a Ranking sheet.
    """
    out_root = EXPERIMENTS_DIR / "RD_CRD_MERGED_ALL"
    out_root.mkdir(parents=True, exist_ok=True)

    suites_map: Dict[str, List[str]] = {}
    results_data: Dict[str, Dict[str, Dict[str, float]]] = {}
    raw_data: Dict[str, Dict[str, List[float]]] = {}

    for tag, dim in SUITES:
        rd_root = _suite_folder("RD", tag, dim)
        crd_root = _suite_folder("CRD", tag, dim)
        funcs = _list_functions(rd_root, crd_root)
        if not funcs:
            print(f"[merge-all] {tag} D={dim}: no overlapping function folders, skip")
            continue
        suite_key = f"{tag}_D{dim}"
        suites_map[suite_key] = funcs

        for func in funcs:
            rd_df = _read_raw_runs(rd_root / func)
            crd_df = _read_raw_runs(crd_root / func)
            if rd_df.empty and crd_df.empty:
                continue
            merged = pd.concat([rd_df, crd_df], ignore_index=True, sort=False)
            merged = merged[merged["Algorithm"].isin(ALGO_ORDER)].copy()
            if merged.empty:
                continue

            results_data[func] = {}
            raw_data[func] = {}
            for algo in [a for a in ALGO_ORDER if a in merged["Algorithm"].unique()]:
                vals = merged[merged["Algorithm"] == algo]["BestFitness"].astype(float).values
                results_data[func][algo] = dict(
                    mean=float(np.mean(vals)),
                    std=float(np.std(vals)),
                    best=float(np.min(vals)),
                    worst=float(np.max(vals)),
                    median=float(np.median(vals)),
                )
                raw_data[func][algo] = vals.tolist()

    if not suites_map:
        print("[merge-all] nothing to merge")
        return

    algo_names_in_data = [a for a in ALGO_ORDER if any(a in r for r in results_data.values())]

    generate_results_excel(results_data, algo_names_in_data, suites_map, str(out_root))
    generate_wilcoxon_excel(raw_data, algo_names_in_data, suites_map, str(out_root))
    generate_friedman_excel(raw_data, algo_names_in_data, suites_map, str(out_root))

    n_funcs = sum(len(v) for v in suites_map.values())
    print(
        f"[merge-all] {len(suites_map)} suites, {n_funcs} functions, "
        f"{len(algo_names_in_data)} algorithms -> {out_root / 'Excel Files'}"
    )


def gather_all_figures() -> None:
    """Copy every per-function PNG from the merged suite folders into one place.

    Output layout::

        experiments/RD_CRD_MERGED_ALL/All Figures/
            Convergence Curves/<func>.png
            Box Plots/<func>.png
    """
    out_root = EXPERIMENTS_DIR / "RD_CRD_MERGED_ALL" / "All Figures"
    out_conv = out_root / "Convergence Curves"
    out_box = out_root / "Box Plots"
    out_conv.mkdir(parents=True, exist_ok=True)
    out_box.mkdir(parents=True, exist_ok=True)

    n_conv = n_box = 0
    for tag, dim in SUITES:
        merged_root = EXPERIMENTS_DIR / f"RD_CRD_MERGED_{tag}_D{dim}"
        if not merged_root.exists():
            continue
        for func_dir in sorted(merged_root.iterdir()):
            if not func_dir.is_dir() or func_dir.name == "Excel Files":
                continue
            for src in (func_dir / "Convergence Curves").glob("*.png"):
                shutil.copy2(src, out_conv / src.name)
                n_conv += 1
            for src in (func_dir / "Box Plots").glob("*.png"):
                shutil.copy2(src, out_box / src.name)
                n_box += 1

    print(
        f"[merge-all] gathered {n_conv} convergence + {n_box} box-plot images "
        f"into {out_root}"
    )


def main():
    for tag, dim in SUITES:
        merge_suite(tag, dim)
    merge_all_suites_into_one_xls()
    gather_all_figures()
    print("[merge] done.")


if __name__ == "__main__":
    main()
