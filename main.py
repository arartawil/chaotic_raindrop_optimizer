"""Entry point that orchestrates setup, experiments, and analysis phases.

Examples:
    python main.py --phase setup
    python main.py --phase experiments
    python main.py --phase analysis
    python main.py --phase all              # everything in order
    python main.py --phase experiments --quick  # smoke test
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
HEURILAB_ROOT = Path("c:/Users/ROG SRTIX/Desktop/pkg/heurilab")
PHAGE_ROOT = Path("c:/Users/ROG SRTIX/Desktop/pkg/Phage Lifecycle")
for p in (PROJECT_ROOT, HEURILAB_ROOT, PHAGE_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


# ── Phase 1: setup ───────────────────────────────────────────────────────────
def phase_setup(verbose: bool = True) -> int:
    """Verify dependencies and import heurilab + Phage CEC2022."""
    fails: list[str] = []
    try:
        import heurilab  # noqa: F401
        if verbose:
            print(f"[setup] heurilab {getattr(heurilab, '__version__', '?')} OK")
    except Exception as exc:
        fails.append(f"heurilab: {exc}")

    try:
        from cec2022 import get_cec2022_functions
        funcs = get_cec2022_functions(dim=10)
        if verbose:
            print(f"[setup] CEC2022: {len(funcs)} functions available at D=10")
    except Exception as exc:
        fails.append(f"cec2022 (Phage Lifecycle): {exc}")

    try:
        from algorithms import RD, CRD_VARIANTS
        if verbose:
            print(f"[setup] CRD variants: {list(CRD_VARIANTS.keys())}")
    except Exception as exc:
        fails.append(f"algorithms: {exc}")

    try:
        import numpy, scipy, matplotlib, pandas, seaborn  # noqa: F401
        from tqdm import tqdm  # noqa: F401
        from openpyxl import Workbook  # noqa: F401
        if verbose:
            print("[setup] numpy / scipy / pandas / matplotlib / seaborn / tqdm / openpyxl OK")
    except Exception as exc:
        fails.append(f"py-deps: {exc}")

    if fails:
        print("[setup] FAILED:")
        for f in fails:
            print(f"        - {f}")
        return 1
    print("[setup] all deps OK")
    return 0


# ── Phase 2: experiments ────────────────────────────────────────────────────
def phase_experiments(extra_args: list[str]) -> int:
    cmd = [sys.executable, str(PROJECT_ROOT / "experiments" / "run_experiments.py"), *extra_args]
    print(f"[experiments] {' '.join(cmd)}")
    return subprocess.call(cmd)


# ── Phase 3: analysis (stats + plots + latex) ───────────────────────────────
def phase_analysis(results_dir: Path, plots_dir: Path, tables_dir: Path) -> int:
    rc = 0
    rc |= subprocess.call([
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "statistical_analysis.py"),
        "--results-dir", str(results_dir),
    ])
    rc |= subprocess.call([
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "generate_plots.py"),
        "--results-dir", str(results_dir),
    ])
    rc |= subprocess.call([
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "generate_latex_tables.py"),
        "--results-dir", str(results_dir),
        "--out-dir", str(tables_dir),
    ])
    return rc


# ── Driver ──────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("setup", "experiments", "analysis", "all"),
                   default="all")
    p.add_argument("--results-dir", default=str(PROJECT_ROOT / "experiments_output"))
    p.add_argument("--plots-dir", default=str(PROJECT_ROOT / "plots"))
    p.add_argument("--tables-dir", default=str(PROJECT_ROOT / "tables"))
    # Pass-through for run_experiments.py
    p.add_argument("--runs", type=int, default=None)
    p.add_argument("--pop", type=int, default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--no-competitors", action="store_true")
    p.add_argument("--no-engineering", action="store_true")
    p.add_argument("--cec2017-only", action="store_true")
    p.add_argument("--cec2022-only", action="store_true")
    p.add_argument("--cec2017-dims", nargs="+", type=int, default=None)
    p.add_argument("--cec2022-dims", nargs="+", type=int, default=None)
    p.add_argument("--cec2017-only-funcs", nargs="+", type=int, default=None)
    p.add_argument("--cec2022-only-funcs", nargs="+", type=int, default=None)
    p.add_argument("--max-procs", type=int, default=None)
    p.add_argument("--crd-subset", nargs="+", default=None)
    return p.parse_args()


def _build_experiment_args(args) -> list[str]:
    out = ["--out-dir", str(args.results_dir)]
    if args.runs is not None:
        out += ["--runs", str(args.runs)]
    if args.pop is not None:
        out += ["--pop", str(args.pop)]
    if args.quick:
        out.append("--quick")
    if args.no_competitors:
        out.append("--no-competitors")
    if args.no_engineering:
        out.append("--no-engineering")
    if args.cec2017_only:
        out.append("--cec2017-only")
    if args.cec2022_only:
        out.append("--cec2022-only")
    if args.cec2017_dims:
        out += ["--cec2017-dims", *map(str, args.cec2017_dims)]
    if args.cec2022_dims:
        out += ["--cec2022-dims", *map(str, args.cec2022_dims)]
    if args.cec2017_only_funcs is not None:
        out += ["--cec2017-only-funcs", *map(str, args.cec2017_only_funcs)]
    if args.cec2022_only_funcs is not None:
        out += ["--cec2022-only-funcs", *map(str, args.cec2022_only_funcs)]
    if args.max_procs is not None:
        out += ["--max-procs", str(args.max_procs)]
    if args.crd_subset:
        out += ["--crd-subset", *args.crd_subset]
    return out


def main():
    args = parse_args()
    rc = 0

    if args.phase in ("setup", "all"):
        rc = phase_setup()
        if rc != 0 and args.phase != "all":
            return rc

    if args.phase in ("experiments", "all"):
        rc = phase_experiments(_build_experiment_args(args))
        if rc != 0 and args.phase != "all":
            return rc

    if args.phase in ("analysis", "all"):
        rc = phase_analysis(
            Path(args.results_dir), Path(args.plots_dir), Path(args.tables_dir)
        )

    return rc


if __name__ == "__main__":
    sys.exit(main())
