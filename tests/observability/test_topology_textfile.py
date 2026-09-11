import json
from pathlib import Path

from profiling_lab.topology_textfile import build_gauges


def test_build_gauges_emits_supplied_components_and_edges(tmp_path: Path) -> None:
    (tmp_path / "storage-topology.json").write_text(json.dumps({
        "components": [{"id": "meta-0", "role": "metadata"}, {"id": "data-0", "role": "data"}],
        "edges": [{"source": "client-0", "destination": "meta-0", "relation": "metadata"}],
    }))

    gauges = build_gauges(tmp_path)

    assert {(g.name, g.labels.get("component")) for g in gauges} >= {
        ("profiling_topology_component_info", "meta-0"),
        ("profiling_topology_component_info", "data-0"),
    }
    assert any(g.name == "profiling_topology_edge_info" and g.labels["relation"] == "metadata" for g in gauges)
