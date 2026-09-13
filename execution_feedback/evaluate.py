"""Extract model code and execute tests in resource-limited Docker workers."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Any

from execution_feedback.common import index_tasks, read_jsonl, write_jsonl


RESULT_PREFIX = "__EXECUTION_FEEDBACK_RESULT__="
FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
RUNNER = r'''import json
from pathlib import Path

PREFIX = "__EXECUTION_FEEDBACK_RESULT__="
result = {"phase": "syntax", "total": 0, "passed": 0, "failures": []}
try:
    source = Path("solution.py").read_text(encoding="utf-8")
    compiled = compile(source, "solution.py", "exec")
except BaseException as exc:
    result["failures"].append(f"{type(exc).__name__}: {exc}")
    print(PREFIX + json.dumps(result))
    raise SystemExit(2)

namespace = {"__name__": "solution"}
try:
    exec(compiled, namespace)
    payload = json.loads(Path("tests.json").read_text(encoding="utf-8"))
    if payload.get("setup"):
        exec(payload["setup"], namespace)
    result["phase"] = "tests"
    result["total"] = len(payload["tests"])
    for ordinal, test in enumerate(payload["tests"]):
        try:
            exec(test, namespace)
            result["passed"] += 1
        except BaseException as exc:
            result["failures"].append(f"test[{ordinal}] {type(exc).__name__}: {exc}")
except BaseException as exc:
    result["phase"] = "runtime"
    result["failures"].append(f"{type(exc).__name__}: {exc}")
print(PREFIX + json.dumps(result))
raise SystemExit(0 if result["total"] > 0 and result["passed"] == result["total"] else 1)
'''


def extract_code(response: str) -> str:
    if not isinstance(response, str) or not response.strip():
        return ""
    matches = FENCE_RE.findall(response)
    return max(matches, key=len).strip() + "\n" if matches else response.strip() + "\n"


def docker_command(work_dir: Path, image: str, memory: str, cpus: float, pids_limit: int, name: str) -> list[str]:
    return [
        "docker", "run", "--rm", "--name", name, "--network", "none", "--read-only",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--memory", memory, "--cpus", str(cpus), "--pids-limit", str(pids_limit),
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m", "--user", "65534:65534",
        "--mount", f"type=bind,src={work_dir.resolve()},dst=/work,readonly",
        "--workdir", "/work", image, "python", "_runner.py",
    ]


def _kill_container(name: str) -> None:
    """subprocess.run(timeout=...) only kills the `docker run` client on
    TimeoutExpired; the container it started keeps running unattended (no
    stop signal reaches it), still holding its full CPU/memory allocation.
    Observed directly: an evaluated `while True: pass` candidate stayed at
    100% CPU indefinitely after being classified as a timeout. --rm only
    removes a container on its own exit, not on the client disconnecting, so
    a forced removal is required. Best-effort: the daemon may already be
    gone, and a failure here must not turn a valid timeout result into a
    worse one.
    """
    try:
        subprocess.run(["docker", "rm", "--force", name], capture_output=True, timeout=10, check=False)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass


def _parse_result(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            try:
                value = json.loads(line[len(RESULT_PREFIX):])
            except json.JSONDecodeError:
                return None
            return value if isinstance(value, dict) else None
    return None


def evaluate_candidate(
    task: dict[str, Any], candidate: dict[str, Any], *, engine: str = "docker",
    image: str = "python:3.12-slim", timeout_seconds: float = 10.0,
    memory: str = "256m", cpus: float = 1.0, pids_limit: int = 64,
) -> dict[str, Any]:
    started = time.perf_counter()
    code = extract_code(candidate.get("response", ""))
    base = {
        "task_id": task["task_id"], "candidate_id": str(candidate.get("candidate_id", "0")),
        "split": task["split"], "source": task["source"], "response": candidate.get("response", ""),
        "code": code, "engine": engine, "image": image if engine == "docker" else None,
    }
    if not code:
        return {**base, "status": "fail", "score": 0.0, "passed_tests": 0, "total_tests": len(task["tests"]), "failure_phase": "extraction", "failures": ["no code found"], "duration_seconds": time.perf_counter() - started}
    with tempfile.TemporaryDirectory(prefix="execution-feedback-") as temporary:
        work_dir = Path(temporary)
        (work_dir / "solution.py").write_text(code, encoding="utf-8")
        (work_dir / "tests.json").write_text(json.dumps({"setup": task.get("test_setup", ""), "tests": task["tests"]}), encoding="utf-8")
        (work_dir / "_runner.py").write_text(RUNNER, encoding="utf-8")
        for path in work_dir.iterdir():
            path.chmod(0o444)
        work_dir.chmod(0o755)
        container_name = f"execution-feedback-{uuid.uuid4().hex}"
        command = docker_command(work_dir, image, memory, cpus, pids_limit, container_name) if engine == "docker" else [sys.executable, "_runner.py"]
        try:
            completed = subprocess.run(command, cwd=None if engine == "docker" else work_dir, text=True, capture_output=True, timeout=timeout_seconds, check=False)
        except subprocess.TimeoutExpired as exc:
            if engine == "docker":
                _kill_container(container_name)
            return {**base, "status": "timeout", "score": 0.0, "passed_tests": 0, "total_tests": len(task["tests"]), "failure_phase": "timeout", "failures": [f"exceeded {timeout_seconds}s"], "stdout": exc.stdout or "", "stderr": exc.stderr or "", "duration_seconds": time.perf_counter() - started}
        except (FileNotFoundError, OSError) as exc:
            return {**base, "status": "infra_error", "score": 0.0, "passed_tests": 0, "total_tests": len(task["tests"]), "failure_phase": "worker", "failures": [f"{type(exc).__name__}: {exc}"], "duration_seconds": time.perf_counter() - started}
    parsed = _parse_result(completed.stdout)
    if engine == "docker" and completed.returncode in {125, 126, 127}:
        status = "infra_error"
    elif parsed is None:
        status = "fail"
    else:
        status = "pass" if completed.returncode == 0 and parsed.get("total") == parsed.get("passed") else "fail"
    reported_total = int(parsed.get("total", 0)) if parsed else 0
    total = reported_total or len(task["tests"])
    passed = int(parsed.get("passed", 0)) if parsed else 0
    return {
        **base, "status": status, "score": passed / total if total else 0.0,
        "passed_tests": passed, "total_tests": total,
        "failure_phase": parsed.get("phase") if parsed and status != "pass" else None,
        "failures": parsed.get("failures", []) if parsed else ["worker did not emit a result"],
        "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-4000:],
        "worker_returncode": completed.returncode, "duration_seconds": time.perf_counter() - started,
    }


def evaluate_file(tasks_path: Path, candidates_path: Path, output_path: Path, **kwargs: Any) -> dict[str, int]:
    tasks = index_tasks(tasks_path)
    candidates = list(read_jsonl(candidates_path))
    seen: set[tuple[str, str]] = set()
    jobs = []
    for candidate in candidates:
        task_id = candidate.get("task_id")
        candidate_id = str(candidate.get("candidate_id", "0"))
        if task_id not in tasks:
            raise ValueError(f"candidate references unknown task: {task_id}")
        key = (task_id, candidate_id)
        if key in seen:
            raise ValueError(f"duplicate candidate: {key}")
        seen.add(key)
        jobs.append((tasks[task_id], candidate))
    workers = int(kwargs.pop("workers", 1))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(lambda job: evaluate_candidate(job[0], job[1], **kwargs), jobs))
    write_jsonl(output_path, results)
    counts = {status: 0 for status in ("pass", "fail", "timeout", "infra_error")}
    for result in results:
        counts[result["status"]] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", choices=("docker", "local"), default="docker")
    parser.add_argument("--image", default="python:3.12-slim")
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--memory", default="256m")
    parser.add_argument("--cpus", type=float, default=1.0)
    parser.add_argument("--pids-limit", type=int, default=64)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.engine == "local":
        print("warning: local execution is for trusted smoke tests only", file=sys.stderr)
    counts = evaluate_file(args.tasks, args.candidates, args.output, engine=args.engine, image=args.image, timeout_seconds=args.timeout_seconds, memory=args.memory, cpus=args.cpus, pids_limit=args.pids_limit, workers=args.workers)
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
