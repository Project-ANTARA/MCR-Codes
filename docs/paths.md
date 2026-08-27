# paths.py

## File Overview
Central path management and Orekit data initialization utility for the ANTARA mission analysis project. All simulation scripts route file I/O through this module to maintain consistent output locations and handle Orekit data setup.

## Dependencies
- **Standard Library**: `pathlib.Path`, `os`
- **External**: `orekit.pyhelpers` (`download_orekit_data_curdir`, `setup_orekit_curdir`)
- **Internal**: None (this is a leaf dependency used by all simulation scripts)

## Exports / Public API

### Module Constants
| Constant | Type | Description |
|----------|------|-------------|
| `SRC_DIR` | `Path` | Absolute path to `src/` directory |
| `PROJECT_ROOT` | `Path` | Absolute path to repository root |
| `DATA_DIR` | `Path` | `PROJECT_ROOT / "data"` - Orekit data and reference inputs |
| `RESULTS_DIR` | `Path` | `PROJECT_ROOT / "results"` - Simulation outputs (gitignored) |
| `FIGURES_DIR` | `Path` | `PROJECT_ROOT / "figures"` - Generated plots (gitignored) |

### Functions

#### `ensure_dir(path: Path) -> Path`
Creates directory (including parents) if it doesn't exist.
- **Params**: `path` - Target directory path
- **Returns**: The same `Path` object (for chaining)
- **Side effects**: Creates directories on filesystem

#### `results_dir(*parts: str) -> Path`
Returns a directory path under `results/`, creating it if needed.
- **Params**: `parts` - Path components to join under `results/`
- **Returns**: `Path` to the ensured directory
- **Usage**: `results_dir("mcr")` → `results/mcr/`

#### `figures_dir(*parts: str) -> Path`
Returns a directory path under `figures/`, creating it if needed.
- **Params**: `parts` - Path components to join under `figures/`
- **Returns**: `Path` to the ensured directory
- **Usage**: `figures_dir("thermal")` → `figures/thermal/`

#### `results_path(*parts: str) -> Path`
Returns a file path under `results/`, ensuring parent directory exists.
- **Params**: `parts` - Path components (last component is filename)
- **Returns**: `Path` to the file (parent directory created)
- **Usage**: `results_path("mcr", "power.csv")` → `results/mcr/power.csv`

#### `figures_path(*parts: str) -> Path`
Returns a file path under `figures/`, ensuring parent directory exists.
- **Params**: `parts` - Path components (last component is filename)
- **Returns**: `Path` to the file (parent directory created)
- **Usage**: `figures_path("thermal", "plot.png")` → `figures/thermal/plot.png`

#### `init_orekit_data(download_if_missing: bool = False) -> Path`
Initializes Orekit data directory for the Orekit VM.
- **Params**: `download_if_missing` - If True, downloads Orekit data automatically
- **Returns**: `Path` to the Orekit data directory that was used
- **Behavior**:
  1. Checks for existing data at `data/orekit/orekit-data/orekit-data-main` or `data/orekit-data.zip`
  2. If found, calls `setup_orekit_curdir()` and returns the path
  3. If not found and `download_if_missing=True`: changes to `DATA_DIR`, downloads data via `download_orekit_data_curdir()`, sets up Orekit, returns zip path
  4. If not found and `download_if_missing=False`: raises `FileNotFoundError`
- **Side effects**: Calls Orekit's `setup_orekit_curdir()`; may download ~500MB data on first run

## Internal Logic
- Uses `Path.resolve()` for absolute, symlink-resolved paths
- `ensure_dir()` is the single mkdir primitive used by all other helpers
- `init_orekit_data()` searches two canonical locations before optionally downloading
- Download temporarily changes CWD to `DATA_DIR` to control where Orekit writes the zip

## Usage Example
```python
from paths import init_orekit_data, results_path, figures_path

# Must call before any Orekit imports in scripts
init_orekit_data(download_if_missing=True)

# Writing simulation outputs
output_csv = results_path("mcr", "power_analysis.csv")
plot_png = figures_path("thermal", "temperature_profile.png")

# Both parent directories are auto-created
```

## Notes / TODOs
- **Orekit data location**: Expects data at `data/orekit/orekit-data/orekit-data-main/` (extracted) or `data/orekit-data.zip`. The zip is auto-downloaded if `download_if_missing=True`.
- **Thread safety**: `init_orekit_data()` changes global CWD temporarily — not thread-safe if called concurrently.
- **Gitignore**: `results/` and `figures/` are gitignored; `data/orekit/` and `data/orekit-data.zip` are also gitignored per `.gitignore`.