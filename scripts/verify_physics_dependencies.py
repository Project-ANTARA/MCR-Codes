"""Check that physics backends load and execute in the Orekit interpreter."""

import argparse
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
import sys

import numpy as np
import orekit
import ppigrf
import spacepy
from spacepy import irbempy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-ci-env", action="store_true")
    args = parser.parse_args()
    print("Python:", sys.executable, sys.version)
    if args.require_ci_env:
        assert "/micromamba/envs/antara/" in sys.executable, sys.executable
    for name, module in (("orekit", orekit), ("spacepy", spacepy), ("ppigrf", ppigrf)):
        print(name, version(name), module.__file__)
        assert Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    orekit.initVM()
    field = np.asarray(ppigrf.igrf(0.0, 0.0, 600.0, datetime(2025, 3, 1)))
    assert field.shape == (3, 1) and np.isfinite(field).all()
    print("IGRF East/North/Up (nT):", field.ravel())
    # Exercise the bundled IRBEM library. This is AP8, NOT an AP9 validation.
    flux = irbempy.get_AEP8(10.0, [1.0, 1.5], model="min", fluxtype="diff", particles="p")
    assert np.isfinite(flux) and flux >= 0.0
    print("IRBEM AP8 differential proton flux:", flux)
    print("AP9 is not provided by get_AEP8; ANTARA retains its configured baseline.")


if __name__ == "__main__":
    main()
