"""Backward-compatible shortcut for the canonical training module.

Prefer: python -m training.train_experiment --actor_arch <architecture>
"""

import sys


ARCHITECTURES = {"mlp", "gnn", "transformer", "gnn_transformer"}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1].lower() not in ARCHITECTURES:
        choices = ", ".join(sorted(ARCHITECTURES))
        print(f"Usage: python train.py <architecture> [extra args]\nChoices: {choices}")
        return 2
    architecture = sys.argv[1].lower()
    from training.config import parse_args
    from training.train_experiment import configure_logging, run_experiment

    config = parse_args(["--actor_arch", architecture, *sys.argv[2:]])
    configure_logging(config.log_level)
    run_experiment(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
