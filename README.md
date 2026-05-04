# Chaotic Raindrop Optimizer (CRD)

Code and experimental results for the paper:

> **"Chaotic Raindrop Optimizer: A Comprehensive Study on the Impact of
> Chaotic Maps for Enhanced Global Optimization"**

The base algorithm is the **Raindrop Optimizer (RD)** (Sci. Rep. 2025,
DOI: 10.1038/s41598-025-15832-w). Twelve chaotic maps replace the random
parameters to produce twelve **CRD** variants. Both the base RD and all
12 CRDs are benchmarked on **CEC2017 (D=30)** and **CEC2020 (D=10)**.

Best-performing variant overall: **CRD-Chebyshev** (Friedman mean rank
≈ 2.62 across 39 functions, beats RD on 25/39 with Wilcoxon p<0.05).

## Repository layout

```
chaotic_raindrop_optimizer/
├── algorithms/                 # Algorithm implementations
│   ├── RD_optimizer.py         # Base Raindrop Optimizer (heurilab._Base subclass)
│   ├── CRD_optimizer.py        # Chaotic RD + 12 ready-made variants
│   └── chaotic_maps.py         # 12 chaotic maps + ChaoticStream RNG
├── experiments/                # Reusable experiment + analysis modules
│   ├── run_experiments.py      # Generic per-function MP runner
│   ├── algorithm_roster.py     # RD + 12 CRDs + 10 competitors roster
│   ├── benchmark_suites.py     # CEC2017/CEC2022 BenchmarkSuite builders
│   ├── statistical_analysis.py # Wilcoxon, Friedman+Nemenyi, Cohen's d
│   ├── generate_plots.py       # Heatmap, CD diagram, radar, win/tie/loss
│   └── generate_latex_tables.py# 5 LaTeX tables per suite
├── scripts/                    # Per-batch convenience runners
│   ├── run_rd_cec2017_*.py     # Base RD on CEC2017 (3 batches: 10/10/9 funcs)
│   ├── run_rd_cec2020_d10.py   # Base RD on all 10 CEC2020 functions
│   ├── run_crd_cec2017_*.py    # 12 CRDs on CEC2017 (5 batches: 10/5/5/5/4 funcs)
│   ├── run_crd_cec2020_d10.py  # 12 CRDs on all 10 CEC2020 functions
│   ├── complete_cec2020_f10.py # Patches partial F10 results
│   └── merge_rd_crd_results.py # Combines RD + CRD outputs into one bundle
├── results/                    # Final merged outputs (committed)
│   ├── excel/                  # 3 workbooks covering CEC2017 + CEC2020
│   │   ├── Results.xlsx        # Mean ± Std, Ranking sheet, "All Functions"
│   │   ├── Wilcoxon.xlsx       # p-values vs RD, W+/W=/W− summary
│   │   └── Friedman.xlsx       # Friedman ranks + Nemenyi post-hoc
│   └── figures/
│       ├── Convergence Curves/ # 39 PNGs, all 13 algorithms per function
│       └── Box Plots/          # 39 PNGs, all 13 algorithms per function
├── main.py                     # setup / experiments / analysis driver
├── PROJECT_OVERVIEW.md         # paper-ready methods/results write-up
├── requirements.txt
├── LICENSE                     # MIT
└── README.md
```

## External dependencies

This project uses the [`heurilab`](https://github.com/arartawil/heurilab)
package as its experiment infrastructure (`_Base` algorithm class,
`run_experiment` runner, CEC2017/CEC2020 suites, CSV/Excel exporters, and
plot helpers). Install it locally and add its path to `sys.path` before
running any script — `main.py` already does this if `heurilab` is checked
out next to this repo.

## Quick start

```powershell
# 1) verify dependencies
python main.py --phase setup

# 2) run every batch (each script multiprocesses one process per function)
python scripts/run_rd_cec2017_first10.py
python scripts/run_rd_cec2017_next10.py
python scripts/run_rd_cec2017_last9.py
python scripts/run_rd_cec2020_d10.py

python scripts/run_crd_cec2017_first10.py
python scripts/run_crd_cec2017_next5.py
python scripts/run_crd_cec2017_next5_v2.py
python scripts/run_crd_cec2017_next5_v3.py
python scripts/run_crd_cec2017_last4.py
python scripts/run_crd_cec2020_d10.py

# 3) merge RD + CRD into one Excel + one figure folder
python scripts/merge_rd_crd_results.py
```

Per-batch output lives under `experiments/<RD|CRD>_<SUITE>_D<dim>/<func>/`
(git-ignored — easy to regenerate). The merged outputs that **are**
committed live under `results/`.

## Algorithms shipped

| Group | Members |
|-------|---------|
| Base | `RD` |
| Chaotic RDs | `CRD-Logistic`, `CRD-Tent`, `CRD-Sine`, `CRD-Singer`, `CRD-Sinusoidal`, `CRD-Chebyshev`, `CRD-Circle`, `CRD-Gauss`, `CRD-Bernoulli`, `CRD-Piecewise`, `CRD-Iterative`, `CRD-LogisticTent` |
| Competitors (optional, via heurilab) | PSO, GWO, WOA, SCA, MFO, HHO, SSA, AOA, SAO, DE |

## Experimental settings

| Setting | Value |
|---------|-------|
| Population size | 30 |
| Iterations | 300 |
| Independent runs | 30 |
| CEC2017 dim | 30 |
| CEC2020 dim | 10 |
| Function-evaluation budget | identical for every algorithm |

## Citation

If you use this code, please cite the paper (forthcoming) and the
underlying RD reference:

* Liu, X., et al. *"Raindrop Optimizer: A novel meta-heuristic algorithm
  for global optimization."* Scientific Reports (2025).
  DOI: 10.1038/s41598-025-15832-w.

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
