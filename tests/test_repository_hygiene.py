"""Lightweight repository checks; do not import or launch simulations."""
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_requirements_have_unique_distribution_names():
    names = []
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", line)
        assert match, f"Expected a package requirement: {line}"
        names.append(re.sub(r"[-_.]+", "-", match.group().lower()))
    assert names
    assert len(names) == len(set(names)), "Duplicate dependency declarations"
    assert "orekit" not in names, "Orekit is provided by Conda, not this pip list"
