import json
from types import SimpleNamespace

import pytest

from megatron_lab.measurement import measure_execution


@pytest.mark.parametrize('fail', [False, True])
def test_timing_preserves_calls_exceptions_and_restores_methods(tmp_path, fail):
    class Manager:
        def save(self, value, callback=None):
            if fail:
                raise ValueError('save failed')
            return value

        def finalize_async_saves(self, state, blocking=False, terminate=False):
            return state, blocking, terminate

    original = Manager.save
    cuda = SimpleNamespace(is_available=lambda: False)
    try:
        with measure_execution(tmp_path, 'train', manager_class=Manager, cuda=cuda):
            assert Manager().finalize_async_saves('state', True, terminate=True) == ('state', True, True)
            assert Manager().save(42) == 42
    except ValueError as exc:
        assert fail and str(exc) == 'save failed'
    assert Manager.save is original
    records = [json.loads(line) for line in (tmp_path / 'measurements/rank-0-train.jsonl').read_text().splitlines()]
    assert records[0]['blocking'] is True and records[0]['terminate'] is True
    assert records[1]['success'] is not fail
    assert records[-1]['success'] is not fail
    assert all(record['seconds'] >= 0 for record in records)
    assert all(record['ended_monotonic_seconds'] >= record['started_monotonic_seconds'] for record in records)
    assert all(record['ended_wall_time_ns'] >= record['started_wall_time_ns'] for record in records)
    assert records[-1]['peak_cuda_allocated_bytes'] is None
