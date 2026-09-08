import json
from pathlib import Path

import pytest

from trl_lab.public_data import PRESETS, adapt_row, select_bounded, write_public_jsonl


def test_native_and_self_oss_adapters_have_canonical_messages() -> None:
    native = adapt_row(PRESETS["no_robots"], {"prompt_id": "n1", "messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A", "category": "ignored"}]})
    code = adapt_row(PRESETS["self_oss"], {"id": 7, "instruction": "Implement it", "response": "Done"})
    assert [message["role"] for message in native["messages"]] == ["user", "assistant"]
    assert code["prompt_id"] == "7"
    assert all(set(message) == {"role", "content"} for message in code["messages"])


def test_xlam_canonicalizes_json_without_tool_execution() -> None:
    row = adapt_row(PRESETS["xlam"], {"id": 3, "query": "Find it", "tools": '[{"name":"lookup","parameters":{"type":"object"}}]', "answers": '[{"name":"lookup","arguments":{"x":1}}]'})
    assert row["messages"][0]["role"] == "system"
    assert '"name":"lookup"' in row["messages"][0]["content"]
    assert json.loads(row["messages"][-1]["content"])[0]["name"] == "lookup"


def test_xlam_malformed_json_fails_instead_of_dropping() -> None:
    with pytest.raises(ValueError, match="tools is not valid JSON"):
        adapt_row(PRESETS["xlam"], {"id": 1, "query": "Q", "tools": "not-json", "answers": "[]"})


def test_native_structured_tool_calls_are_rejected() -> None:
    with pytest.raises(ValueError, match="structured tool calls"):
        adapt_row(PRESETS["no_robots"], {"prompt_id": "x", "messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A", "tool_calls": []}]})


def test_bounded_selection_is_deterministic_and_disjoint() -> None:
    rows = [{"id": i, "instruction": f"Q{i}", "response": f"A{i}"} for i in range(100)]
    first = select_bounded(rows, PRESETS["self_oss"], 8, 3, 42, 100)
    second = select_bounded(rows, PRESETS["self_oss"], 8, 3, 42, 100)
    assert first == second
    assert not ({row["prompt_id"] for row in first[0]} & {row["prompt_id"] for row in first[1]})


def test_duplicate_prompts_with_distinct_source_ids_are_grouped() -> None:
    rows = [{"id": 1, "instruction": "same prompt", "response": "A"}, {"id": 2, "instruction": "same prompt", "response": "B"}]
    with pytest.raises(ValueError, match="bounded stream"):
        select_bounded(rows, PRESETS["self_oss"], 1, 1, 42, 2)


def test_eval_quota_does_not_reassign_train_rows() -> None:
    rows = [{"id": i, "instruction": f"Q{i}", "response": f"A{i}"} for i in range(200)]
    first_train, _ = select_bounded(rows, PRESETS["self_oss"], 8, 3, 42, 200)
    second_train, _ = select_bounded(rows, PRESETS["self_oss"], 8, 8, 42, 200)
    assert [row["prompt_id"] for row in first_train] == [row["prompt_id"] for row in second_train]


def test_manifest_records_revision_hashes_and_source_provenance(tmp_path: Path) -> None:
    train = [adapt_row(PRESETS["self_oss"], {"id": 1, "instruction": "Q", "response": "A"})]
    evaluation = [adapt_row(PRESETS["self_oss"], {"id": 2, "instruction": "Q2", "response": "A2"})]
    manifest = write_public_jsonl(tmp_path, PRESETS["self_oss"], "a" * 40, train, evaluation, 42, 10)
    assert manifest["dataset_revision"] == "a" * 40
    assert manifest["files"]["training"]["sha256"]
    assert json.loads((tmp_path / "training.jsonl").read_text())["provenance"]["adapter_version"]


def test_swe_trajectories_are_reference_only() -> None:
    with pytest.raises(ValueError, match="reference-only"):
        adapt_row(PRESETS["swe_trajectories"], {"instance_id": "x"})
