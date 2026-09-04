"""Validate the local Prometheus and Grafana monitoring control plane."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from run_summary import make_run_summary


TARGET_FILES = ("applications.json", "gpus.json", "nodes.json")


def validate_target_files(directory: Path) -> dict[str, int]:
    """Validate Prometheus file-discovery targets and return their counts."""
    missing = [name for name in TARGET_FILES if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"missing target files: {', '.join(missing)}")

    group_count = 0
    target_count = 0
    for name in TARGET_FILES:
        path = directory / name
        try:
            groups = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in {path}: {error}") from error
        if not isinstance(groups, list):
            raise ValueError(f"{path} must contain a JSON array")
        for index, group in enumerate(groups):
            if not isinstance(group, dict):
                raise ValueError(f"{path} group {index} must be an object")
            targets = group.get("targets")
            labels = group.get("labels", {})
            if not isinstance(targets, list) or not all(
                isinstance(target, str) and target.strip() for target in targets
            ):
                raise ValueError(
                    f"{path} group {index} must contain non-empty string targets"
                )
            if not isinstance(labels, dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in labels.items()
            ):
                raise ValueError(f"{path} group {index} labels must be strings")
            group_count += 1
            target_count += len(targets)

    return {
        "files": len(TARGET_FILES),
        "groups": group_count,
        "targets": target_count,
    }


def summarize_prometheus_targets(payload: dict[str, Any]) -> dict[str, int]:
    """Count active Prometheus targets by reported health."""
    if payload.get("status") != "success":
        raise ValueError("Prometheus target response did not report success")
    targets = payload.get("data", {}).get("activeTargets")
    if not isinstance(targets, list):
        raise ValueError("Prometheus target response is missing activeTargets")
    up = sum(target.get("health") == "up" for target in targets)
    return {"configured": len(targets), "up": up, "down": len(targets) - up}


def grafana_database_is_healthy(payload: dict[str, Any]) -> bool:
    """Return whether Grafana reports a healthy database."""
    return payload.get("database") == "ok"


def _request(url: str, timeout: float = 5.0) -> bytes:
    request = Request(url, headers={"User-Agent": "sft-lab-observability-validation/1"})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def _read_json(url: str) -> dict[str, Any]:
    value = json.loads(_request(url).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{url} did not return a JSON object")
    return value


def _wait_until_ready(url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            _request(url)
            return
        except (HTTPError, URLError, TimeoutError) as error:
            last_error = error
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    raise RuntimeError(f"endpoint did not become ready: {url}: {last_error}")


def validate_stack(args: argparse.Namespace) -> int:
    """Validate live endpoints, write a summary, and return a process status."""
    target_files = validate_target_files(args.target_dir)
    started = time.monotonic()
    errors: list[str] = []
    prometheus_ready = False
    grafana_ready = False
    grafana_database_ok = False
    prometheus_query_ok = False
    target_status = {"configured": 0, "up": 0, "down": 0}

    try:
        _wait_until_ready(
            f"{args.prometheus_url.rstrip('/')}/-/ready",
            args.timeout,
        )
        prometheus_ready = True
    except (RuntimeError, ValueError) as error:
        errors.append(str(error))

    try:
        grafana_health_url = f"{args.grafana_url.rstrip('/')}/api/health"
        _wait_until_ready(grafana_health_url, args.timeout)
        grafana_ready = True
        grafana_database_ok = grafana_database_is_healthy(
            _read_json(grafana_health_url)
        )
        if not grafana_database_ok:
            errors.append("Grafana database health is not ok")
    except (RuntimeError, ValueError, HTTPError, URLError) as error:
        errors.append(str(error))

    if prometheus_ready:
        try:
            target_status = summarize_prometheus_targets(
                _read_json(
                    f"{args.prometheus_url.rstrip('/')}/api/v1/targets?state=active"
                )
            )
            query = urlencode({"query": "up"})
            query_result = _read_json(
                f"{args.prometheus_url.rstrip('/')}/api/v1/query?{query}"
            )
            prometheus_query_ok = query_result.get("status") == "success"
            if not prometheus_query_ok:
                errors.append("Prometheus test query did not report success")
        except (ValueError, HTTPError, URLError) as error:
            errors.append(str(error))

    targets_acceptable = not args.require_targets_up or (
        target_status["configured"] > 0 and target_status["down"] == 0
    )
    if not targets_acceptable:
        errors.append("one or more configured Prometheus targets are not up")

    stack_valid = all(
        (
            prometheus_ready,
            grafana_ready,
            grafana_database_ok,
            prometheus_query_ok,
            targets_acceptable,
        )
    )
    elapsed = time.monotonic() - started
    summary = make_run_summary(
        configuration={
            "prometheus_url": args.prometheus_url,
            "grafana_url": args.grafana_url,
            "target_dir": str(args.target_dir),
            "require_targets_up": args.require_targets_up,
        },
        environment={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "docker": os.environ.get("VALIDATION_DOCKER_VERSION"),
            "docker_compose": os.environ.get("VALIDATION_COMPOSE_VERSION"),
        },
        performance={"validation_seconds": elapsed},
        artifacts={
            "summary_file": str(args.output),
            "target_dir": str(args.target_dir),
        },
        validation={
            "target_files_valid": True,
            "target_files": target_files,
            "prometheus_ready": prometheus_ready,
            "prometheus_query_ok": prometheus_query_ok,
            "grafana_ready": grafana_ready,
            "grafana_database_ok": grafana_database_ok,
            "prometheus_targets": target_status,
            "stack_valid": stack_valid,
            "errors": errors,
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[validation] summary={args.output}")
    print(
        f"[validation] stack_valid={str(stack_valid).lower()} "
        f"targets={target_status['configured']} "
        f"up={target_status['up']} down={target_status['down']}"
    )
    for error in errors:
        print(f"[validation] error={error}")
    return 0 if stack_valid else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    targets = subparsers.add_parser(
        "check-targets",
        help="validate Prometheus file-discovery target files",
    )
    targets.add_argument("--target-dir", type=Path, required=True)

    stack = subparsers.add_parser(
        "validate-stack",
        help="validate running Prometheus and Grafana services",
    )
    stack.add_argument("--target-dir", type=Path, required=True)
    stack.add_argument("--prometheus-url", default="http://127.0.0.1:9090")
    stack.add_argument("--grafana-url", default="http://127.0.0.1:3000")
    stack.add_argument("--output", type=Path, required=True)
    stack.add_argument("--timeout", type=float, default=60.0)
    stack.add_argument("--require-targets-up", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "check-targets":
            result = validate_target_files(args.target_dir)
            print(
                f"[targets] files={result['files']} groups={result['groups']} "
                f"targets={result['targets']}"
            )
            return 0
        return validate_stack(args)
    except ValueError as error:
        print(f"[validation] error={error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
