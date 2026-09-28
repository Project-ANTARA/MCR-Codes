# Support direct execution and qualified simulations.* imports.
import sys as _sys
from pathlib import Path as _Path
if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from typing import Literal, Tuple
from simulations.helper.antara_models import MissionConfig


def build_scenario(name: Literal["leop_24h", "no_leop_24h", "leop_1yr"]) -> Tuple[MissionConfig, bool]:
    """Build a predefined scenario configuration.

    Returns:
        Tuple of (MissionConfig, skip_leop_flag)
    """
    if name == "leop_24h":
        return (
            MissionConfig(
                sim_duration_hours=24.0,
                output_tag="leop_24h",
                telemetry_flush_interval_hours=6.0,
            ),
            False,
        )
    elif name == "no_leop_24h":
        return (
            MissionConfig(
                sim_duration_hours=24.0,
                output_tag="no_leop_24h",
                telemetry_flush_interval_hours=6.0,
            ),
            True,
        )
    elif name == "leop_1yr":
        return (
            MissionConfig(
                sim_duration_hours=8760.0,
                output_tag="leop_1yr",
                telemetry_flush_interval_hours=6.0,
                coarse_step_sec=20.0,
            ),
            False,
        )
    else:
        raise ValueError(f"Unknown scenario: {name}")
