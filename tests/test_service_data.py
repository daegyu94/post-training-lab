import json

import pytest

from sft_lab.prepare_service_data import prepare_dataset


def record(trace_id: str, session_id: str, split: str = "train") -> dict:
    return {
        "trace_id": trace_id,
        "session_id": session_id,
        "split": split,
        "messages": [
            {"role": "user", "content": f"Question {trace_id}"},
            {"role": "assistant", "content": "Reviewed answer"},
        ],
        "review": {
            "status": "approved",
            "label_source": "expert",
            "sensitive_data_removed": True,
        },
    }


def write_source(path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_prepare_dataset_separates_training_and_test_answers(tmp_path) -> None:
    source = tmp_path / "traces.jsonl"
    output = tmp_path / "prepared"
    write_source(source, [record("train-1", "session-1"), record("test-1", "session-2", "test")])

    manifest = prepare_dataset(source, output)

    train = json.loads((output / "training.jsonl").read_text().strip())
    test = json.loads((output / "test.jsonl").read_text().strip())
    assert train["messages"][-1]["content"] == "Reviewed answer"
    assert test["prompt"][-1]["role"] == "user"
    assert test["reference_answer"] == "Reviewed answer"
    assert manifest["counts"] == {"train": 1, "validation": 0, "test": 1}


def test_prepare_dataset_excludes_unreviewed_rows(tmp_path) -> None:
    source = tmp_path / "traces.jsonl"
    output = tmp_path / "prepared"
    row = record("one", "session-1")
    row["review"]["status"] = "pending"
    write_source(source, [row])

    manifest = prepare_dataset(source, output)

    assert manifest["excluded"] == {"not_approved": 1}
    assert (output / "training.jsonl").read_text() == ""


def test_prepare_dataset_rejects_cross_split_session(tmp_path) -> None:
    source = tmp_path / "traces.jsonl"
    write_source(source, [record("one", "same", "train"), record("two", "same", "test")])

    with pytest.raises(ValueError, match="spans train and test"):
        prepare_dataset(source, tmp_path / "prepared")
