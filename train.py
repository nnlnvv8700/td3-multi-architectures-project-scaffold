"""Backward-compatible shortcut for the canonical training module.

Prefer: python -m training.train_experiment --actor_arch <architecture>
"""

import subprocess
import sys


ARCHITECTURES = {"mlp", "gnn", "transformer", "gnn_transformer"}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1].lower() not in ARCHITECTURES:
        choices = ", ".join(sorted(ARCHITECTURES))
        print(f"Usage: python train.py <architecture> [extra args]\nChoices: {choices}")
        return 2
    architecture = sys.argv[1].lower()
    command = [
        sys.executable,
        "-m",
        "training.train_experiment",
        "--actor_arch",
        architecture,
        *sys.argv[2:],
    ]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
