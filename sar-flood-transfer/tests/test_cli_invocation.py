"""Regression test for the Colab notebook's actual invocation pattern.

The scripts under src/training/ use absolute `from src.xxx import ...`
imports, which only resolve when the project root is on sys.path. Running
`python src/training/foo.py` does NOT put it there (Python adds the
script's own directory instead) and fails with ModuleNotFoundError; running
`python -m src.training.foo` does. The notebook must use the latter form --
this test catches a regression to the former (which broke a real cloud run).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI_MODULES = (
    "src.training.pretrain_oilspill",
    "src.training.run_matrix",
    "src.training.frozen_probe",
    "src.training.train_flood_segmentation",
)


def test_cli_modules_runnable_via_dash_m():
    for module in CLI_MODULES:
        result = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{module} --help failed:\n{result.stderr}"
        assert "ModuleNotFoundError" not in result.stderr


def test_cli_modules_fail_as_direct_script_path_documents_the_footgun():
    # Not something to "fix" here -- this documents WHY the notebook must
    # use -m, by confirming the naive invocation still fails the way it did
    # in the real Colab session that surfaced this bug.
    script = PROJECT_ROOT / "src" / "training" / "pretrain_oilspill.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "ModuleNotFoundError" in result.stderr
