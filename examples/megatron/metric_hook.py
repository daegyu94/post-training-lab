"""Example hook that exports already-computed Megatron metrics per rank."""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Mapping

from profiling_lab.prometheus_textfile import GaugeSample, write_gauges


def export_megatron_step(
    *,
    iteration: int,
    step_time_seconds: float,
    tokens_per_second: float,
    timer_seconds: Mapping[str, float],
) -> Path:
    """Call at Megatron's normal logging interval after reducing desired timers."""
    rank = os.environ.get("RANK", "0")
    labels = {
        "run_id": os.environ["PROFILE_RUN_ID"],
        "node": socket.gethostname(),
        "rank": rank,
        "local_rank": os.environ.get("LOCAL_RANK", "0"),
        "tp_rank": os.environ.get("TP_RANK", "unknown"),
        "pp_rank": os.environ.get("PP_RANK", "unknown"),
        "dp_rank": os.environ.get("DP_RANK", "unknown"),
    }
    samples = [
        GaugeSample("llm_training_iteration", "Current training iteration.", iteration, labels),
        GaugeSample("llm_training_step_time_seconds", "Latest logged step time.", step_time_seconds, labels),
        GaugeSample("llm_training_tokens_per_second", "Latest logged token throughput.", tokens_per_second, labels),
    ]
    for timer_name, value in timer_seconds.items():
        samples.append(
            GaugeSample(
                "llm_megatron_timer_seconds",
                "Megatron timer value for the latest logging interval.",
                value,
                labels | {"timer": timer_name},
            )
        )
    return write_gauges(
        Path(os.environ.get("NODE_EXPORTER_TEXTFILE_DIR", "/var/lib/node_exporter/textfile_collector")),
        f"llm_megatron_rank_{rank}.prom",
        samples,
    )
