from __future__ import annotations

from pathlib import Path

from orekit.pyhelpers import download_orekit_data_curdir, setup_orekit_curdir


SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_dir(*parts: str) -> Path:
    return ensure_dir(RESULTS_DIR.joinpath(*parts))


def figures_dir(*parts: str) -> Path:
    return ensure_dir(FIGURES_DIR.joinpath(*parts))


def results_path(*parts: str) -> Path:
    path = RESULTS_DIR.joinpath(*parts)
    ensure_dir(path.parent)
    return path


def figures_path(*parts: str) -> Path:
    path = FIGURES_DIR.joinpath(*parts)
    ensure_dir(path.parent)
    return path


def init_orekit_data(download_if_missing: bool = False) -> Path:
    candidates = [
        PROJECT_ROOT / "orekit-data" / "orekit-data-main",
        PROJECT_ROOT / "orekit-data.zip",
    ]

    for candidate in candidates:
        if candidate.exists():
            setup_orekit_curdir(str(candidate))
            return candidate

    if not download_if_missing:
        raise FileNotFoundError(
            "Orekit data not found under data/. Expected "
            "'data/orekit/orekit-data/orekit-data-main' or 'data/orekit-data.zip'."
        )

    original_cwd = Path.cwd()
    ensure_dir(DATA_DIR)
    try:
        import os

        os.chdir(DATA_DIR)
        download_orekit_data_curdir()
    finally:
        os.chdir(original_cwd)

    orekit_zip = DATA_DIR / "orekit-data.zip"
    setup_orekit_curdir(str(orekit_zip))
    return orekit_zip
