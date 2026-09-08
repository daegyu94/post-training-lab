import json
import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize('node_rank', [0, 1])
def test_comparison_reads_last_rank_logs_on_their_node(tmp_path, node_rank):
    calls = tmp_path / 'calls.jsonl'
    fake = tmp_path / 'python'
    fake.write_text('#!/usr/bin/env python3\nimport sys,json\n' +
                    f'with open({str(calls)!r}, "a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n')
    fake.chmod(0o755)
    env = os.environ | dict(PYTHON=str(fake), NODE_RANK=str(node_rank), NNODES='2',
                           NPROC_PER_NODE='1', MASTER_ADDR='example',
                           OUTPUT_DIR=str(tmp_path / 'out'))
    subprocess.run(['bash', 'scripts/run_spark_cluster.sh'],
                   cwd=Path(__file__).parents[2] / 'backends' / 'megatron',
                   env=env, check=True, capture_output=True)
    compared = [args for args in map(json.loads, calls.read_text().splitlines())
                if 'megatron_lab.compare' in args]
    assert len(compared) == node_rank
    if node_rank:
        args = compared[0]
        assert args[args.index('--base-log') + 1].endswith('rank-1-base.log')
        assert args[args.index('--tuned-log') + 1].endswith('rank-1-tuned.log')
        assert args[args.index('--metadata') + 1].endswith('tuned-rank-1.json')
