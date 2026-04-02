import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _mean(values: List[float]) -> float:
    valid = [v for v in values if v == v]
    if not valid:
        return float("nan")
    return sum(valid) / len(valid)


def _std(values: List[float]) -> float:
    valid = [v for v in values if v == v]
    if len(valid) <= 1:
        return 0.0 if valid else float("nan")
    m = _mean(valid)
    return (sum((v - m) ** 2 for v in valid) / len(valid)) ** 0.5


def collect_eval_history(results_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for csv_path in results_root.rglob("eval_history.csv"):
        run_dir = csv_path.parent.parent
        with csv_path.open("r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                row["run_dir"] = str(run_dir)
                rows.append(row)
    return rows


def aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("task_level", "task1"),
            row.get("actor_arch", "unknown"),
            row.get("deterministic", "True"),
            row.get("checkpoint_step", "0"),
            row.get("eval_split", "eval"),
        )
        buckets[key].append(row)

    out: List[Dict[str, Any]] = []
    metric_keys = [
        "success_rate",
        "reward_mean",
        "final_error_mean",
        "rmse_mean",
        "max_deviation_mean",
        "path_len_exec_mean",
        "path_efficiency_mean",
        "avg_jerk_mean",
        "collision_rate",
        "min_clearance_mean",
    ]

    for key, bucket in buckets.items():
        task_level, actor_arch, deterministic, checkpoint_step, eval_split = key
        rec: Dict[str, Any] = {
            "task_level": task_level,
            "actor_arch": actor_arch,
            "deterministic": deterministic,
            "checkpoint_step": checkpoint_step,
            "eval_split": eval_split,
            "n_runs": len(bucket),
        }

        for mk in metric_keys:
            vals = [_safe_float(row.get(mk, float("nan"))) for row in bucket]
            rec[f"{mk}_mean"] = _mean(vals)
            rec[f"{mk}_std"] = _std(vals)

        out.append(rec)

    out.sort(key=lambda x: (x["task_level"], x["actor_arch"], int(float(x["checkpoint_step"]))))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate eval_history.csv across seeds")
    parser.add_argument("--results_root", type=str, default="./results", help="Root directory containing run folders")
    parser.add_argument("--output", type=str, default="./results/seed_aggregate.csv", help="Output CSV path")
    parser.add_argument("--output_json", type=str, default="./results/seed_aggregate.json", help="Output JSON path")
    args = parser.parse_args()

    root = Path(args.results_root)
    rows = collect_eval_history(root)
    aggregated = aggregate(rows)

    out_csv = Path(args.output)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if aggregated:
        fieldnames = list(aggregated[0].keys())
        with out_csv.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(aggregated)

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as fp:
        json.dump(aggregated, fp, ensure_ascii=False, indent=2)

    print(f"[Aggregate] collected rows: {len(rows)}")
    print(f"[Aggregate] groups: {len(aggregated)}")
    print(f"[Aggregate] csv: {out_csv}")
    print(f"[Aggregate] json: {out_json}")


if __name__ == "__main__":
    main()
