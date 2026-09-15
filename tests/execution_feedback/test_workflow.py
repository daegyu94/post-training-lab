from __future__ import annotations

import json
from pathlib import Path

import pytest

from execution_feedback.common import index_tasks, read_jsonl, write_jsonl
from execution_feedback.compare import compare
from execution_feedback.evaluate import _kill_container, docker_command, evaluate_candidate, extract_code
from execution_feedback.feedback import build_feedback
from execution_feedback.generate import candidate_budgets
from execution_feedback.prepare import adapt_mbpp, prepare
from execution_feedback.train import dpo_needs_precomputed_ref_logps


def test_prepare_writes_disjoint_splits_and_sft_files(tmp_path: Path) -> None:
    tasks = tuple(
        {
            "task_id": f"mbpp-{index}", "prompt": "Write a function.",
            "reference_solution": "def f(): return 1", "tests": ["assert f() == 1"],
            "split": split, "template_group": f"mbpp-{index}",
        }
        for index, split in enumerate(("train", "validation", "test"), 1)
    )
    manifest = prepare(tasks, tmp_path, "google-research-datasets/mbpp", "a" * 40, "cc-by-4.0")
    assert manifest["counts"] == {"train": 1, "validation": 1, "test": 1}
    tasks = index_tasks(tmp_path / "tasks.jsonl")
    assert len(tasks) == 3
    train_ids = {row["task_id"] for row in read_jsonl(tmp_path / "train.jsonl")}
    test_ids = {row["task_id"] for row in read_jsonl(tmp_path / "test.jsonl")}
    assert train_ids.isdisjoint(test_ids)
    assert all(row["messages"][-1]["role"] == "assistant" for row in read_jsonl(tmp_path / "initial_sft_train.jsonl"))


def test_adapt_mbpp_reads_the_sanitized_configs_actual_field_names() -> None:
    # The "sanitized" MBPP config uses 'prompt' for the problem text, not
    # 'text' (that was the bug: real data has no 'text' field, so this raised
    # KeyError on every row before hitting Docker or a model at all).
    row = {
        "source_file": "Benchmark Questions Verification V2.ipynb", "task_id": 602,
        "prompt": "Write a python function to find the first repeated character in a given string.",
        "code": "def first_repeated_char(str1):\n    return str1[0]",
        "test_imports": [], "test_list": ['assert first_repeated_char("abcabc") == "a"'],
    }
    task = adapt_mbpp(row, "train")
    assert task["task_id"] == "mbpp-602"
    assert task["prompt"].startswith("Write a python function")
    assert task["reference_solution"] == row["code"]
    assert task["tests"] == row["test_list"]


def test_extract_code_prefers_largest_python_fence() -> None:
    response = "Explanation\n```python\ndef answer():\n    return 42\n```\n```py\nx = 1\n```"
    assert extract_code(response) == "def answer():\n    return 42\n"


def test_candidate_budgets_add_real_truncated_samples_without_id_collisions() -> None:
    assert candidate_budgets(2, 200, 1, 32) == [("0", 200), ("1", 200), ("truncated-0", 32)]


def test_docker_command_applies_isolation(tmp_path: Path) -> None:
    command = docker_command(tmp_path, "python:3.12-slim", "256m", 1.0, 64, "execution-feedback-test")
    joined = " ".join(command)
    assert "--network none" in joined
    assert "--read-only" in command
    assert "--cap-drop ALL" in joined
    assert "no-new-privileges" in joined
    assert "readonly" in joined
    assert "--name execution-feedback-test" in joined


def test_kill_container_is_best_effort(monkeypatch) -> None:
    # A TimeoutExpired-killed `docker run` client leaves its container running
    # unattended (observed directly: a `while True: pass` candidate stayed at
    # 100% CPU indefinitely after being classified "timeout"). The forced
    # `docker rm` cleanup must not raise even if the daemon is unreachable --
    # a failed cleanup must not turn a valid timeout result into a crash.
    seen = []
    monkeypatch.setattr(
        "execution_feedback.evaluate.subprocess.run",
        lambda *args, **kwargs: seen.append(args[0]) or (_ for _ in ()).throw(FileNotFoundError("no docker")),
    )
    _kill_container("execution-feedback-test")
    assert seen == [["docker", "rm", "--force", "execution-feedback-test"]]


def test_local_evaluator_reports_pass_and_partial_failure() -> None:
    task = {
        "task_id": "one", "prompt": "implement", "reference_solution": "", "source": "test", "split": "train",
        "tests": ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"], "test_setup": "",
    }
    passed = evaluate_candidate(task, {"candidate_id": "p", "response": "def add(a, b):\n    return a + b"}, engine="local")
    failed = evaluate_candidate(task, {"candidate_id": "f", "response": "def add(a, b):\n    return 3"}, engine="local")
    assert passed["status"] == "pass" and passed["score"] == 1.0
    assert failed["status"] == "fail" and failed["score"] == 0.5


def test_local_evaluator_ignores_candidate_builtin_exec_patch() -> None:
    task = {
        "task_id": "patch", "prompt": "implement", "reference_solution": "", "source": "test", "split": "train",
        "tests": ["assert add(1, 2) == 3"], "test_setup": "",
    }
    response = "import builtins\nbuiltins.exec = lambda *args: None\ndef add(a, b):\n    return 0"
    result = evaluate_candidate(task, {"candidate_id": "patch", "response": response}, engine="local")
    assert result["status"] == "fail"


def test_feedback_uses_same_train_pool_and_excludes_timeout(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    evaluations = tmp_path / "evaluations.jsonl"
    task = {"task_id": "one", "prompt": "implement", "reference_solution": "def f(): pass", "source": "test", "split": "train", "tests": ["assert f() == 1"], "test_setup": ""}
    write_jsonl(tasks, [task])
    write_jsonl(evaluations, [
        {"task_id": "one", "candidate_id": "p", "split": "train", "status": "pass", "score": 1.0, "code": "def f(): return 1"},
        {"task_id": "one", "candidate_id": "f", "split": "train", "status": "fail", "score": 0.0, "code": "def f(): return 0"},
        {"task_id": "one", "candidate_id": "t", "split": "train", "status": "timeout", "score": 0.0, "code": "while True: pass"},
    ])
    manifest = build_feedback(tasks, evaluations, tmp_path / "feedback")
    assert manifest["filtered_sft_count"] == 1
    assert manifest["dpo_pair_count"] == 1
    pair = next(read_jsonl(tmp_path / "feedback" / "dpo.jsonl"))
    assert pair["provenance"]["chosen_candidate_id"] == "p"
    assert pair["provenance"]["rejected_candidate_id"] == "f"
    assert pair["provenance"]["candidate_pool_sha256"]


def test_dpo_precomputes_ref_logps_only_for_full_fine_tuning() -> None:
    # ref_model=None + no PEFT makes DPOTrainer reload a whole second copy of the
    # model to keep as reference; PEFT models (fresh --lora-r or a resumed
    # adapter checkpoint) already avoid that via adapter-disable.
    assert dpo_needs_precomputed_ref_logps(is_adapter=False, lora_r=0) is True
    assert dpo_needs_precomputed_ref_logps(is_adapter=True, lora_r=0) is False
    assert dpo_needs_precomputed_ref_logps(is_adapter=False, lora_r=8) is False


def test_feedback_rejects_non_train_rows(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    evaluations = tmp_path / "evaluations.jsonl"
    write_jsonl(tasks, [{"task_id": "one", "prompt": "p", "reference_solution": "r", "source": "test", "split": "test", "tests": ["assert True"], "test_setup": ""}])
    write_jsonl(evaluations, [{"task_id": "one", "candidate_id": "0", "split": "test", "status": "pass", "score": 1.0, "code": "x=1"}])
    with pytest.raises(ValueError, match="train-only"):
        build_feedback(tasks, evaluations, tmp_path / "feedback")


def test_compare_requires_identical_test_tasks(tmp_path: Path) -> None:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    write_jsonl(a, [{"task_id": "one", "candidate_id": "0", "split": "test", "status": "pass", "duration_seconds": 1.0}])
    write_jsonl(b, [{"task_id": "one", "candidate_id": "0", "split": "test", "status": "fail", "duration_seconds": 2.0}])
    result = compare({"A": a, "B": b})
    assert result["variants"]["A"]["pass_at_1"] == 1.0
    assert result["variants"]["B"]["pass_at_1"] == 0.0


def test_manifest_is_valid_json(tmp_path: Path) -> None:
    prepare([{
        "task_id": "mbpp-1", "prompt": "Write a function.",
        "reference_solution": "def f(): return 1", "tests": ["assert f() == 1"],
        "split": "train", "template_group": "mbpp-1",
    }], tmp_path, "google-research-datasets/mbpp", "a" * 40, "cc-by-4.0")
    json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))


def test_timeout_output_is_json_serializable_and_cleans_container(monkeypatch) -> None:
    import subprocess
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if command[1] == "run":
            raise subprocess.TimeoutExpired(command, 1, output=b"progress\n", stderr=b"\xff")
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr("execution_feedback.evaluate.subprocess.run", run)
    task = {"task_id": "timeout", "split": "train", "source": "test", "tests": ["assert True"]}
    result = evaluate_candidate(task, {"response": "while True: pass"}, timeout_seconds=1)
    assert result["status"] == "timeout"
    assert result["stdout"] == "progress\n"
    json.dumps(result)
    assert calls[1][:3] == ["docker", "rm", "--force"]


@pytest.mark.parametrize("payload", [
    {"total": 0, "passed": 0},
    {"total": 2, "passed": 2},
    {"total": "one", "passed": "one"},
    {"total": 1, "passed": -1},
])
def test_invalid_grade_cannot_pass(monkeypatch, payload) -> None:
    import subprocess
    from execution_feedback.evaluate import RESULT_PREFIX
    monkeypatch.setattr("execution_feedback.evaluate.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, RESULT_PREFIX + json.dumps(payload), ""))
    task = {"task_id": "grade", "split": "test", "source": "test", "tests": ["assert True"]}
    result = evaluate_candidate(task, {"response": "pass"})
    assert result["status"] == "fail"
