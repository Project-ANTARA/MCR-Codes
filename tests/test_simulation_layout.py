"""Relocation and compatibility checks; never import annual studies."""
import ast
import os
from pathlib import Path
import pickle
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
GROUPS = {
    "ditl": ["ANTARA_Science_Sim", "compare_scenarios", "scenarios", "plot_fsm_diagram"],
    "legacy": ["ANTARA_Master"],
    "thermal": ["ThermalSimulator"],
    "conops": ["ANTARA_conops"],
    "power": ["ANTARA_MCR", "ANTARA_avg", "ANTARA_FINAL_POWERGEN", "TradeCase"],
    "detumble": ["leop_detumble_mc", "analyze_detumble_mc"],
    "orbit": ["ANTARA_drag_finalconfig", "ANTARA_drag_study"],
    "pointing": ["ANTARA_pointing_budget"],
    "helper": ["paths", "antara_models", "fsm_spec", "slew_profile", "wheel_dynamics", "telemetry_metrics"],
}


def test_sources_are_moved_not_copied():
    for group, names in GROUPS.items():
        for name in names:
            assert not (SRC / f"{name}.py").exists()
            ast.parse((SRC / "simulations" / group / f"{name}.py").read_text())
    for group in GROUPS:
        tree = ast.parse((SRC / "simulations" / group / "__init__.py").read_text())
        assert all(isinstance(node, ast.Expr) for node in tree.body)


def test_module_guides_are_colocated():
    for group, names in GROUPS.items():
        for name in names:
            assert not (ROOT / "docs" / f"{name}.md").exists()
        assert (SRC / "simulations" / group / "README.md").is_file()
    for name in ("fsm_diagram.png", "fsm_diagram.svg", "fsm_diagram.mmd"):
        assert (SRC / "simulations" / "ditl" / name).is_file()
        assert not (ROOT / "docs" / name).exists()


def test_legacy_pickle_class_reference_resolves():
    from io import BytesIO
    from simulations.helper.serialization import load_trusted_pickle
    from simulations.detumble.leop_detumble_mc import DetumbleTrialResult
    from simulations.helper.antara_models import MissionConfig
    assert load_trusted_pickle(BytesIO(b"cleop_detumble_mc\nDetumbleTrialResult\n.")) is DetumbleTrialResult
    assert load_trusted_pickle(BytesIO(b"cantara_models\nMissionConfig\n.")) is MissionConfig
    assert pickle.loads(pickle.dumps(DetumbleTrialResult)) is DetumbleTrialResult


def test_helper_paths_keep_repository_root():
    import orekit
    orekit.initVM()
    from simulations.helper import paths
    assert paths.PROJECT_ROOT == ROOT
    assert paths.SRC_DIR == SRC
    assert paths.RESULTS_DIR == ROOT / "results"
    assert paths.DATA_DIR == ROOT / "data"
    assert paths.FIGURES_DIR == ROOT / "figures"


def test_categorized_spawn_matches_serial(monkeypatch):
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial
    import multiprocessing
    import numpy as np
    from simulations.detumble import leop_detumble_mc as mc
    config = mc.MissionConfig()
    serial = mc.run_monte_carlo(2, config, timeout_sec=1., seed=42, n_workers=1, show_progress=False)
    monkeypatch.setattr(mc, "ProcessPoolExecutor", partial(ProcessPoolExecutor,
                        mp_context=multiprocessing.get_context("spawn")))
    parallel = mc.run_monte_carlo(2, config, timeout_sec=1., seed=42, n_workers=2, show_progress=False)
    assert len(serial) == len(parallel) == 2
    for a, b in zip(serial, parallel):
        np.testing.assert_array_equal(a.rate0_deg_sec, b.rate0_deg_sec)
        assert a.failure_reason == b.failure_reason
        assert a.success == b.success


@pytest.mark.parametrize("relative", [
    "simulations/ditl/ANTARA_Science_Sim.py",
    "simulations/thermal/ThermalSimulator.py",
    "simulations/detumble/leop_detumble_mc.py",
    "simulations/detumble/analyze_detumble_mc.py",
    "simulations/ditl/compare_scenarios.py",
    "simulations/ditl/plot_fsm_diagram.py",
])
def test_cli_help_without_pythonpath(relative, tmp_path):
    env = dict(os.environ, SPACEPY=str(tmp_path), MPLCONFIGDIR=str(tmp_path))
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, str(SRC / relative), "--help"],
                            cwd=ROOT, env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout.lower()
