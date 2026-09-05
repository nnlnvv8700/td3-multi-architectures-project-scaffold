"""Legacy GNN launcher using the canonical architecture and training defaults.

The old banner described unimplemented topology and learning-rate changes.
This entry point preserves its actual behavior: GNN with 500,000 training steps.
Additional CLI options override those defaults.
"""

import subprocess
import sys
from pathlib import Path


def main():
    command = [
        sys.executable,
        "-m",
        "training.train_experiment",
        "--actor_arch",
        "gnn",
        "--max_timesteps",
        "500000",
        *sys.argv[1:],
    ]
    return subprocess.run(command, cwd=Path(__file__).resolve().parent, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
