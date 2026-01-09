# Global CP cs CC-CP cs SCC-CP (Simulation)

This repository runs R simulations comparing three conformal prediction methos for multi-class classification:
- **Globcal CP (GCP)** : one global threshold.
- **Class-Clustered CP (CC-CP)** : cluster-specific thresholds based on label clustering. CC-CP follows the class-conditional conformal prediction method of Ding et al. (NeurIPS 2023), designed for many-class classification problems.

- **Shrinkage Class-Clustered CP (SCC_CP)** : cluster thresholds thrunk toward the global threshold using a selection split.

## Structure
- `R/`: core functions (`methods.R`, `sim.R`)
- `scripts/`: entry points
  - `01_run_grid.R`: run simulations and save `out/*.rds`
  - `02_plot_all.R`: read `out/*.rds` and generate figures/tables
- `out/`: saved results (ignored by git by default)
- `fig/`: figures (ignored by git by default)
- `table/`: LaTeX tables (tracked)

## How to run
```r
source("scripts/01_run_grid.R")
source("scripts/02_plot_all.R")
```

# References
Ding, T., Angelopoulos, A. N., Bates, S., Jordan, M. I., and Tibshirani, R. J. (2023).
*Class-Conditional Conformal Prediction with Many Classes*.
Advances in Neural Information Processing Systems (NeurIPS 2023).
