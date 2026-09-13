"""Exercise registration input handoff without sudo or a real GitHub token."""

from pathlib import Path
import re
import subprocess


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/install_local_runner.sh"


def test_installer_shell_syntax():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_registration_token_reaches_config_environment(tmp_path):
    source = SCRIPT.read_text()
    body = re.search(r"env PATH=/usr/local/bin:/usr/bin:/bin bash -c '(.*?)'", source, re.S).group(1)
    body = body.replace("cd /var/lib/antara-actions/runner", 'cd "$1"')
    config = tmp_path / "config.sh"
    config.write_text('#!/bin/bash\n[[ "$ACTIONS_RUNNER_INPUT_TOKEN" == test-only-token ]]\n')
    config.chmod(0o700)
    result = subprocess.run(["bash", "-c", body, "registration-test", str(tmp_path)],
                            input="test-only-token\n", text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "test-only-token" not in result.stdout + result.stderr
