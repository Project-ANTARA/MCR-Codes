#!/usr/bin/env python3
"""
Generate FSM diagram from SpacecraftFSM.ALLOWED_TRANSITIONS.

Renders with graphviz (PNG/SVG) and emits Mermaid stateDiagram-v2 text.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fsm_spec import ALLOWED_TRANSITIONS, MODE_COLORS, FSM_SPEC


def build_mermaid() -> str:
    """Generate Mermaid stateDiagram-v2 text."""
    lines = [
        "stateDiagram-v2",
        "    [*] --> COLD_AND_DARK",
        "",
    ]

    # Add states with colors (as comments for reference)
    for mode, color in MODE_COLORS.items():
        lines.append(f"    state {mode}")

    # Add transitions
    for from_mode, to_modes in ALLOWED_TRANSITIONS.items():
        for to_mode in sorted(to_modes):
            # Find matching guard from FSM_SPEC
            guard_label = find_guard_label(from_mode, to_mode)
            if guard_label:
                lines.append(f'    {from_mode} --> {to_mode} : {guard_label}')
            else:
                lines.append(f'    {from_mode} --> {to_mode}')

    lines.append("")
    lines.append("    %% Colors (for reference)")
    for mode, color in MODE_COLORS.items():
        lines.append(f"    %% {mode}: {color}")

    return "\n".join(lines)


def find_guard_label(from_mode: str, to_mode: str) -> str:
    """Find the first matching guard from FSM_SPEC for a transition."""
    # Look for guards that result in this specific transition
    for guard_name, description, result_mode in FSM_SPEC:
        if result_mode == to_mode or result_mode == "target_mode":
            # Check if guard description implies this from_mode
            # This is heuristic - we match based on typical patterns
            if from_mode in description or from_mode == "current_mode":
                return f"{guard_name}: {description}"
        if result_mode == "current_mode" and from_mode != "SAFE":
            # Returns current_mode (no transition)
            return f"{guard_name}: {description} (hold)"

    # Special cases
    special_labels = {
        ("COLD_AND_DARK", "DETUMBLE"): "cold_to_detumble",
        ("DETUMBLE", "SUN_POINTING"): "detumble_complete",
        ("DETUMBLE", "SAFE"): "critical_battery / combined_fault",
        ("DETUMBLE", "SLEWING"): "slew_to_SAFE/SUN_POINTING",
        ("SAFE", "SUN_POINTING"): "safe_exit / recovery",
        ("SAFE", "RELAY"): "recovery -> RELAY",
        ("SAFE", "DETECTION"): "recovery -> DETECTION",
        ("SAFE", "SLEWING"): "recovery -> SLEWING",
        ("SLEWING", "SAFE"): "slew_critical_battery / fault",
        ("SLEWING", "SUN_POINTING"): "slew_complete",
        ("SLEWING", "RELAY"): "slew_complete -> RELAY",
        ("SLEWING", "DETECTION"): "slew_complete -> DETECTION",
        ("SUN_POINTING", "SAFE"): "critical_battery / saturation / pointing_budget",
        ("SUN_POINTING", "RELAY"): "relay_priority",
        ("SUN_POINTING", "DETECTION"): "detection_priority",
        ("SUN_POINTING", "SLEWING"): "mode_change requires SLEWING",
        ("DETECTION", "RELAY"): "relay_priority",
        ("DETECTION", "SUN_POINTING"): "low_power / end_dwell",
        ("DETECTION", "SAFE"): "critical_battery / saturation / dwell",
        ("DETECTION", "SLEWING"): "mode_change requires SLEWING",
        ("RELAY", "SUN_POINTING"): "low_power / end_dwell",
        ("RELAY", "SAFE"): "critical_battery / saturation / dwell",
        ("RELAY", "DETECTION"): "detection_priority",
        ("RELAY", "SLEWING"): "mode_change requires SLEWING",
    }

    return special_labels.get((from_mode, to_mode), "")


def render_graphviz(output_dir: Path, format: str = "png"):
    """Render diagram using graphviz."""
    try:
        from graphviz import Digraph
    except ImportError:
        print("graphviz not installed. Skipping PNG/SVG generation.")
        return False

    dot = Digraph("ANTARA_FSM", format=format)
    dot.attr(rankdir="LR", fontsize="12", fontname="DejaVu Sans")
    dot.attr("node", shape="box", style="filled", fontname="DejaVu Sans", fontsize="10")
    dot.attr("edge", fontname="DejaVu Sans", fontsize="9")

    # Add nodes with colors
    for mode, color in MODE_COLORS.items():
        label = mode.replace("_", "\n")
        dot.node(mode, label=label, fillcolor=color, fontcolor="white" if mode in ["COLD_AND_DARK", "SAFE", "SLEWING", "DETECTION", "RELAY"] else "black")

    # Add edges with labels
    for from_mode, to_modes in ALLOWED_TRANSITIONS.items():
        for to_mode in sorted(to_modes):
            label = find_guard_label(from_mode, to_mode)
            if label:
                dot.edge(from_mode, to_mode, label=label, fontsize="8")
            else:
                dot.edge(from_mode, to_mode)

    output_path = output_dir / f"fsm_diagram.{format}"
    dot.render(str(output_path.with_suffix("")), cleanup=True)
    print(f"Saved {format.upper()} to {output_path}")
    return True


def write_mermaid(output_dir: Path):
    """Write Mermaid diagram to file."""
    mermaid = build_mermaid()
    output_path = output_dir / "fsm_diagram.mmd"
    with open(output_path, "w") as f:
        f.write(mermaid)
    print(f"Saved Mermaid to {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate FSM diagram")
    parser.add_argument("--output-dir", type=Path, default=Path("docs"), help="Output directory")
    parser.add_argument("--format", choices=["png", "svg", "both"], default="both", help="Output format")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write Mermaid (always)
    write_mermaid(output_dir)

    # Render graphviz
    if args.format in ["png", "both"]:
        render_graphviz(output_dir, "png")
    if args.format in ["svg", "both"]:
        render_graphviz(output_dir, "svg")

    print(f"\nDone. Outputs in {output_dir}")


if __name__ == "__main__":
    main()