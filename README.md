# ANTARA Mission Analysis Workspace

This workspace has been cleaned into a single working structure so the active simulation code, generated outputs, archived variants, and large Orekit assets are no longer mixed together in the project root.

## Layout

- `src/`: active Python entrypoints and shared path utilities
- `data/`: Orekit datasets and reference inputs
- `results/`: generated CSV/TXT outputs from simulations
- `figures/`: generated plots and images
- `notebooks/`: notebooks kept for ad hoc analysis
- `archive/`: older redundant code variants and MATLAB-era files
- `tmp/`: scratch space

## Main Entry Points

- `python src/ANTARA_Master.py`
- `python src/ANTARA_MCR.py`
- `python src/ANTARA_avg.py`
- `python src/ANTARA_conops.py`

Supporting studies remain in `src/` and now resolve shared data/output paths through `src/paths.py`.

## Notes

- Orekit data is expected under `data/orekit/orekit-data/orekit-data-main` or `data/orekit-data.zip`.
- Historical generated files from earlier runs were preserved and moved into `results/historical_*` and `figures/historical`.
- Older duplicated power-generation variants were archived instead of deleted to avoid losing potentially useful provenance.
