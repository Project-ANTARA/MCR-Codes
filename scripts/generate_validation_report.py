#!/usr/bin/env python3
"""
Generate VALIDATION_REPORT.md from all validation test results.

Runs all six tests/validation/test_*.py modules, captures pass/fail + printed values,
and writes VALIDATION_REPORT.md with one section per subsystem.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

VALIDATION_DIR = Path(__file__).parent.parent / "tests" / "validation"
TESTS = [
    ("test_orbit_propagation.py", "Orbit Propagation", "Skyfield/SGP4 vs Orekit", "period ±1s, RAAN precession ±5%"),
    ("test_igrf_field.py", "IGRF Magnetic Field", "NOAA IGRF-13 calculator", "field magnitude ±5%"),
    ("test_adcs_detumble_time.py", "ADCS Detumble Time", "B-dot τ ≈ I/(k·B²)", "order-of-magnitude"),
    ("test_eps_power_budget.py", "EPS Power Budget", "BOL cell × packing factor", "±2%"),
    ("test_saa_flux_model.py", "SAA/Flux Model", "AP9 quiet-time at L~1.5", "factor of 2 per channel"),
    ("test_comms_link_budget.py", "Comms Link Budget", "Friis link budget", "heuristic within 30% at 45°"),
]


def run_test(script_name: str) -> dict:
    """Run a single validation test and return results."""
    script_path = VALIDATION_DIR / script_name
    if not script_path.exists():
        return {
            "script": script_name,
            "status": "NOT_FOUND",
            "error": f"File not found: {script_path}",
        }

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=VALIDATION_DIR.parent.parent,  # run from repo root for imports
        )
        return {
            "script": script_name,
            "status": "PASS" if result.returncode == 0 else "FAIL",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            "script": script_name,
            "status": "TIMEOUT",
            "error": "Test timed out after 300s",
        }
    except Exception as e:
        return {
            "script": script_name,
            "status": "ERROR",
            "error": str(e),
        }


def load_json_result(test_name: str) -> dict:
    """Load JSON result file if it exists."""
    json_path = VALIDATION_DIR / f"validation_results_{test_name.replace('test_', '').replace('.py', '')}.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    return {}


def main():
    output_dir = Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RUNNING ALL VALIDATION TESTS")
    print("=" * 70)

    results = []
    for script, title, reference, tolerance in TESTS:
        print(f"\nRunning {script}...")
        result = run_test(script)
        json_data = load_json_result(script.replace(".py", ""))
        result["json_data"] = json_data
        results.append((title, reference, tolerance, result))

    # Generate report
    print("\n" + "=" * 70)
    print("GENERATING VALIDATION_REPORT.md")
    print("=" * 70)

    md_lines = [
        "# ANTARA Subsystem Validation Report",
        "",
        f"Generated: {__import__('datetime').datetime.now().isoformat()}",
        f"Repository: ANTARA CubeSat Mission Simulation",
        "",
        "## Summary",
        "",
        "| Subsystem | Status | Reference | Tolerance |",
        "|-----------|--------|-----------|-----------|",
    ]

    overall_pass = True
    for title, reference, tolerance, result in results:
        status = result["status"]
        if status == "PASS":
            status_md = "✅ PASS"
        elif status == "FAIL":
            status_md = "❌ FAIL"
            overall_pass = False
        elif status == "NOT_FOUND":
            status_md = "⚠️ NOT FOUND"
            overall_pass = False
        elif status == "TIMEOUT":
            status_md = "⏱️ TIMEOUT"
            overall_pass = False
        elif status == "ERROR":
            status_md = "💥 ERROR"
            overall_pass = False
        else:
            status_md = f"❓ {status}"
            overall_pass = False

        md_lines.append(f"| {title} | {status_md} | {reference} | {tolerance} |")

    md_lines.extend([
        "",
        f"**Overall: {'✅ ALL PASS' if overall_pass else '❌ SOME FAILED'}**",
        "",
        "## Detailed Results",
        "",
    ])

    for title, reference, tolerance, result in results:
        md_lines.extend([
            f"### {title}",
            "",
            f"**Script:** `{result['script']}`",
            f"**Status:** {result['status']}",
            f"**Reference:** {reference}",
            f"**Tolerance:** {result.get('tolerance', 'N/A')}",
            "",
        ])

        if result.get("stdout"):
            md_lines.extend([
                "#### Stdout",
                "```",
                result["stdout"][:2000] + ("..." if len(result["stdout"]) > 2000 else ""),
                "```",
                "",
            ])

        if result.get("stderr"):
            md_lines.extend([
                "#### Stderr",
                "```",
                result["stderr"][:1000] + ("..." if len(result["stderr"]) > 1000 else ""),
                "```",
                "",
            ])

        if result.get("json_data"):
            md_lines.extend([
                "#### Quantitative Results",
                "```json",
                json.dumps(result["json_data"], indent=2)[:2000] + ("..." if len(json.dumps(result["json_data"])) > 2000 else ""),
                "```",
                "",
            ])

    md_lines.extend([
        "## Notes",
        "",
        "- Tolerances are subsystem-specific and documented in each test.",
        "- AP9 flux models carry ~factor 2-3 intrinsic uncertainty (limited measurements).",
        "- B-dot detumble analytical model is order-of-magnitude due to simplified B-field.",
        "- Comms heuristic (sqrt(sin(el))) is a PDR-level approximation; full link budget needed for CDR.",
        "- Period comparison uses theoretical circular orbit at 600km; real orbit has J2 perturbations.",
        "",
        "---",
        "",
        "*Report generated by `scripts/generate_validation_report.py`*",
    ])

    report_path = output_dir / "VALIDATION_REPORT.md"
    with open(report_path, "w") as f:
        f.write("\n".join(md_lines))

    print(f"\nReport saved to {report_path}")
    print(f"Overall: {'ALL PASS' if overall_pass else 'SOME FAILED'}")

    return overall_pass


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)