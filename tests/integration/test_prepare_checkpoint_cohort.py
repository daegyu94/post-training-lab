import json
from pathlib import Path

import pytest

from experiments.prepare_checkpoint_cohort import build_cohort


def _source(path: Path) -> None:
    path.mkdir()
    (path / "manifest.json").write_text(json.dumps({"dataset": "d", "dataset_revision": "r", "seed": 42}))
    for name, count in (("training", 6), ("validation", 3)):
        (path / f"{name}.jsonl").write_text("".join(
            json.dumps({"prompt_id": f"{name}-{index}", "tokens": index + 1}) + "\n"
            for index in range(count)
        ))


def test_build_cohort_filters_by_rendered_length_and_records_distribution(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _source(source)

    result = build_cohort(
        source, tmp_path / "cohort", dataset_id="d", dataset_revision="r", model_revision="m",
        max_length=4, train_count=3, eval_count=2,
        renderer=lambda row: {"prompt_tokens": row["tokens"], "supervised_tokens": 1},
    )

    assert result["train_count"] == 3
    assert result["length_distribution"]["training"]["maximum"] == 7
    assert result["train_prompt_ids"] == ["training-0", "training-1", "training-2"]
    assert json.loads((tmp_path / "cohort/manifest.json").read_text())["files"]["training"]["sha256"]


def test_build_cohort_refuses_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _source(source)
    output = tmp_path / "cohort"
    output.mkdir()

    with pytest.raises(FileExistsError, match="reuse"):
        build_cohort(
            source, output, dataset_id="d", dataset_revision="r", model_revision="m",
            max_length=4, train_count=1, eval_count=1,
            renderer=lambda row: {"prompt_tokens": 1, "supervised_tokens": 1},
        )
