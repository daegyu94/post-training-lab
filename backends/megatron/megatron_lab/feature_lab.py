"""Feature-lab matrix and result comparison helpers.

The feature runs deliberately use a small dense model when testing optimizer,
checkpoint, or overlap behavior.  Their evidence must not be reported as a
30B MoE training result.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path


FEATURE_VARIANTS = {
    "checkpoint": ("sync", "async"),
    "overlap-grad-reduce": ("off", "on"),
    "recompute": ("full", "selective"),
    "sequence-parallel": ("off", "on"),
    "expert-parallel": ("ep1", "ep2"),
    "lora-ratio": ("ratio-0.1pct", "ratio-0.5pct", "ratio-1pct"),
}

_STEP_RE = re.compile(
    r"\biteration\s+(?P<iteration>\d+)\s*/\s*(?P<total>\d+)"
    r".*?elapsed time per iteration \(ms\):\s*(?P<elapsed>[^|\s]+)"
    r".*?lm loss:\s*(?P<loss>[^|\s]+)"
    r".*?grad norm:\s*(?P<grad>[^|\s]+)"
    r".*?number of skipped iterations:\s*(?P<skipped>\d+)"
    r".*?number of nan iterations:\s*(?P<nan>\d+)"
)
_CHECKPOINT_RE = re.compile(
    r"save-checkpoint\s+\.*:\s*\((?P<minimum>[^,]+),\s*(?P<maximum>[^)]+)\)"
)


@dataclass(frozen=True)
class FeatureRun:
    feature: str
    variant: str
    run_index: int
    warmup: bool
    model_scope: str = "small-dense-feature-model"
    precision: str = "bf16"
    seed: int = 42
    max_length: int = 2048
    micro_batch_size: int = 1
    global_batch_size: int = 8


def validate_feature_variant(feature: str, variant: str) -> None:
    try:
        variants = FEATURE_VARIANTS[feature]
    except KeyError as exc:
        raise ValueError(f"unknown feature: {feature}") from exc
    if variant not in variants:
        raise ValueError(f"{feature} variant must be one of {variants}: {variant}")


def make_feature_run(
    feature: str,
    variant: str,
    run_index: int,
    *,
    warmup: bool = False,
    seed: int = 42,
) -> FeatureRun:
    validate_feature_variant(feature, variant)
    return FeatureRun(feature=feature, variant=variant, run_index=run_index, warmup=warmup, seed=seed)


def summarize_timings(records: list[dict[str, object]]) -> dict[str, object]:
    """Summarize repeats while preserving the raw, non-claiming evidence."""

    measured = [record for record in records if not bool(record.get("warmup"))]
    by_variant: dict[str, list[float]] = {}
    for record in measured:
        variant = str(record["variant"])
        by_variant.setdefault(variant, []).append(float(record["elapsed_seconds"]))
    return {
        "measured_repeats": {
            variant: {
                "count": len(values),
                "median_seconds": statistics.median(values),
                "raw_seconds": values,
            }
            for variant, values in sorted(by_variant.items())
        },
        "interpretation": "descriptive timings only; no speedup claim without a matched runtime result",
    }


def parse_megatron_log(
    log_path: Path,
    *,
    feature: str,
    variant: str,
    run_index: int,
    warmup_steps: int = 2,
    exit_code: int | None = None,
    completed: bool = False,
) -> dict[str, object]:
    """Parse one raw Megatron rank log into bounded, non-claiming evidence."""

    validate_feature_variant(feature, variant)
    if run_index < 0:
        raise ValueError("run_index must be non-negative")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if completed and exit_code != 0:
        raise ValueError("completed log evidence requires process exit code 0")
    raw = log_path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    steps: list[dict[str, object]] = []
    declared_total: int | None = None
    for line in text.splitlines():
        match = _STEP_RE.search(line)
        if not match:
            continue
        total = int(match.group("total"))
        if declared_total is None:
            declared_total = total
        elif total != declared_total:
            raise ValueError("Megatron log contains inconsistent iteration totals")
        step = {
            "step_id": int(match.group("iteration")),
            "elapsed_ms": float(match.group("elapsed")),
            "loss": float(match.group("loss")),
            "grad_norm": float(match.group("grad")),
            "skipped_iterations": int(match.group("skipped")),
            "nan_iterations": int(match.group("nan")),
        }
        if not all(math.isfinite(float(step[field])) for field in ("elapsed_ms", "loss", "grad_norm")):
            raise ValueError(f"non-finite metric at iteration {step['step_id']}")
        if float(step["elapsed_ms"]) <= 0:
            raise ValueError(f"non-positive elapsed time at iteration {step['step_id']}")
        if float(step["grad_norm"]) < 0:
            raise ValueError(f"negative grad norm at iteration {step['step_id']}")
        if step["skipped_iterations"] < 0 or step["nan_iterations"] < 0:
            raise ValueError(f"negative skip/nan count at iteration {step['step_id']}")
        if step["skipped_iterations"] or step["nan_iterations"]:
            raise ValueError(
                f"iteration {step['step_id']} has skipped/nan updates; "
                "benchmark timing is invalid"
            )
        steps.append(step)
    if not steps:
        raise ValueError(f"no Megatron iteration records found in {log_path}")
    step_ids = [int(step["step_id"]) for step in steps]
    if len(step_ids) != len(set(step_ids)):
        raise ValueError("Megatron log contains duplicate iteration records")
    expected_total = declared_total or max(step_ids)
    expected_ids = list(range(1, expected_total + 1))
    if step_ids != expected_ids:
        missing = sorted(set(expected_ids) - set(step_ids))
        raise ValueError(f"Megatron log has missing iterations: {missing}")
    if warmup_steps >= len(steps):
        raise ValueError("warmup_steps must leave at least one steady-state iteration")

    checkpoint_timings = []
    for match in _CHECKPOINT_RE.finditer(text):
        minimum = float(match.group("minimum"))
        maximum = float(match.group("maximum"))
        if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum < 0 or maximum < minimum:
            raise ValueError("invalid save-checkpoint timer")
        checkpoint_timings.append({"min_ms": minimum, "max_ms": maximum})
    steady = steps[warmup_steps:]
    return {
        "feature": feature,
        "variant": variant,
        "run_index": run_index,
        "log_path": str(log_path),
        "raw_log_sha256": hashlib.sha256(raw).hexdigest(),
        "process_exit_code": exit_code,
        # Iteration lines may precede a later checkpoint or teardown failure;
        # completion must be established by an external run-level check.
        "completion_unverified": not completed,
        "completion_verified": completed,
        "warmup_steps": warmup_steps,
        "warmup_step_ids": [int(step["step_id"]) for step in steps[:warmup_steps]],
        "steady_step_ids": [int(step["step_id"]) for step in steady],
        "steps": steps,
        "steady_step_median_ms": statistics.median(float(step["elapsed_ms"]) for step in steady),
        "checkpoint_save_rank_range_ms": checkpoint_timings,
        "interpretation": (
            "descriptive rank-log timing only; checkpoint timers are separate "
            "rank ranges and no speedup or quality claim is generated"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature", choices=tuple(FEATURE_VARIANTS))
    parser.add_argument("--variant")
    parser.add_argument("--run-index", type=int)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--record-glob")
    parser.add_argument("--log", type=Path, help="raw Megatron rank log to parse")
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--exit-code", type=int, help="optional external process exit code evidence")
    parser.add_argument(
        "--completed",
        action="store_true",
        help="mark a parsed log complete only after the launcher returned exit code 0",
    )
    parser.add_argument("--model-id")
    parser.add_argument("--model-revision")
    parser.add_argument("--dataset-id")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--git-commit")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--elapsed-seconds", type=float)
    args = parser.parse_args()

    if args.log:
        if not args.feature or not args.variant or args.run_index is None:
            parser.error("--log requires explicit --feature, --variant, and --run-index")
        result = parse_megatron_log(
            args.log,
            feature=args.feature,
            variant=args.variant,
            run_index=args.run_index,
            warmup_steps=args.warmup_steps,
            exit_code=args.exit_code,
            completed=args.completed,
        )
    elif args.record_glob:
        records = [
            json.loads(Path(path).read_text(encoding="utf-8"))
            for path in sorted(glob.glob(args.record_glob))
        ]
        result = summarize_timings(records)
    elif args.records:
        records = json.loads(args.records.read_text(encoding="utf-8"))
        result = summarize_timings(records)
    else:
        if not args.feature or not args.variant:
            parser.error("--feature and --variant are required without --records")
        result = asdict(
            make_feature_run(
                args.feature,
                args.variant,
                args.run_index if args.run_index is not None else 0,
                warmup=args.warmup,
                seed=args.seed,
            )
        )
        if args.elapsed_seconds is not None:
            result["elapsed_seconds"] = args.elapsed_seconds
        result["process_exit_code"] = args.exit_code
        result["provenance"] = {
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "dataset_id": args.dataset_id,
            "dataset_revision": args.dataset_revision,
            "git_commit": args.git_commit,
            "metadata": str(args.metadata) if args.metadata else None,
            "evidence": str(args.evidence) if args.evidence else None,
        }
    encoded = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
