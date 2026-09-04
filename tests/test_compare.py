from pathlib import Path

import pytest

from megatron_lab.compare import read_last_loss


def test_read_last_loss_uses_final_validation(tmp_path: Path) -> None:
    log = tmp_path / "eval.log"
    log.write_text(
        "validation loss at iteration 0 | lm loss value: 1.500000E+00 |\n"
        "validation loss at iteration 5 | lm loss value: 9.000000E-01 |\n",
        encoding="utf-8",
    )
    assert read_last_loss(log) == pytest.approx(0.9)


def test_read_last_loss_rejects_missing_metric(tmp_path: Path) -> None:
    log = tmp_path / "eval.log"
    log.write_text("no metric\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no validation loss"):
        read_last_loss(log)
