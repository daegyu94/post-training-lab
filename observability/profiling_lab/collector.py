"""Authenticated local telemetry collector and web viewer."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Callable


HOST_FIELDS = {
    "cpu_utilization_percent",
    "memory_used_gib",
    "memory_total_gib",
    "memory_available_gib",
    "swap_used_gib",
    "swap_total_gib",
    "nic_receive_gbps",
    "nic_transmit_gbps",
    "interval_seconds",
}
FRAMEWORK_FIELDS = {
    "training_loss",
    "training_step_time_seconds",
    "training_tokens_per_second",
}
IDENTIFIER = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
TIMER = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
VIEWER = Path(__file__).resolve().parents[1] / "web" / "telemetry.html"
WEB_ROOT = VIEWER.parent.resolve()


def _web_file(request_path: str) -> tuple[bytes, str] | None:
    path = request_path.partition("?")[0]
    if path == "/":
        path = "/telemetry.html"
    candidate = (WEB_ROOT / path.lstrip("/")).resolve()
    if not candidate.is_relative_to(WEB_ROOT) or not candidate.is_file():
        return None
    content_type = {
        ".css": "text/css; charset=utf-8",
        ".html": "text/html; charset=utf-8",
        ".js": "text/javascript; charset=utf-8",
        ".json": "application/json",
    }.get(candidate.suffix, "application/octet-stream")
    return candidate.read_bytes(), content_type


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def validate_host_sample(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict) or set(data) != {"run_id", "node", "metrics"}:
        raise ValueError("expected run_id, node, metrics")
    if any(not isinstance(data[key], str) or not IDENTIFIER.fullmatch(data[key]) for key in ("run_id", "node")):
        raise ValueError("invalid identifier")
    metrics = data["metrics"]
    if not isinstance(metrics, dict) or set(metrics) != HOST_FIELDS:
        raise ValueError("invalid metric fields")
    if any(not _number(value) for value in metrics.values()):
        raise ValueError("metrics must be finite nonnegative numbers")
    if (
        metrics["cpu_utilization_percent"] > 100
        or metrics["interval_seconds"] <= 0
        or metrics["memory_total_gib"] <= 0
        or metrics["memory_used_gib"] > metrics["memory_total_gib"]
        or metrics["memory_available_gib"] > metrics["memory_total_gib"]
        or metrics["swap_used_gib"] > metrics["swap_total_gib"]
    ):
        raise ValueError("invalid metric range")
    return data


def validate_framework_sample(data: Any) -> dict[str, Any]:
    expected = {
        "schema_version", "run_id", "framework", "node", "rank",
        "local_rank", "step", "observed_at", "metrics", "timers",
    }
    if not isinstance(data, dict) or set(data) != expected or data["schema_version"] != 1:
        raise ValueError("invalid framework sample fields")
    if any(not isinstance(data[key], str) or not IDENTIFIER.fullmatch(data[key]) for key in ("run_id", "node")):
        raise ValueError("invalid identifier")
    if data["framework"] not in {"trl", "megatron"}:
        raise ValueError("invalid framework")
    if any(type(data[key]) is not int or data[key] < 0 for key in ("rank", "local_rank", "step")):
        raise ValueError("invalid rank or step")
    if not _number(data["observed_at"]):
        raise ValueError("invalid observation time")
    metrics, timers = data["metrics"], data["timers"]
    if not isinstance(metrics, dict) or not metrics or set(metrics) - FRAMEWORK_FIELDS:
        raise ValueError("invalid framework metrics")
    if not isinstance(timers, dict) or len(timers) > 32 or any(not TIMER.fullmatch(key) for key in timers):
        raise ValueError("invalid framework timers")
    if any(not _number(value) for value in [*metrics.values(), *timers.values()]):
        raise ValueError("framework metrics must be finite nonnegative numbers")
    return data


def make_server(address: tuple[str, int], database: Path, token: str) -> ThreadingHTTPServer:
    database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE IF NOT EXISTS samples (id INTEGER PRIMARY KEY, received_at TEXT, run_id TEXT, node TEXT, metrics TEXT)")
        db.execute("CREATE TABLE IF NOT EXISTS framework_samples (id INTEGER PRIMARY KEY, received_at TEXT, sample TEXT)")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def reply(self, status: int, body: bytes | dict[str, Any], content_type: str = "application/json") -> None:
            payload = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def authorized(self) -> bool:
            return hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {token}")

        def do_GET(self) -> None:
            if self.path == "/healthz":
                return self.reply(200, {"status": "ok"})
            if self.path not in {"/api/telemetry", "/api/framework-metrics"}:
                static = _web_file(self.path)
                return self.reply(200, *static) if static else self.reply(404, {"error": "not found"})
            if not self.authorized():
                return self.reply(401, {"error": "unauthorized"})
            with sqlite3.connect(database) as db:
                if self.path == "/api/telemetry":
                    rows = db.execute("SELECT received_at, run_id, node, metrics FROM samples ORDER BY id DESC LIMIT 1000").fetchall()
                    samples = [dict(received_at=stamp, run_id=run_id, node=node, metrics=json.loads(metrics)) for stamp, run_id, node, metrics in rows]
                    body = {"source": "linux-procfs", "synthetic": False, "samples": samples}
                else:
                    rows = db.execute("SELECT received_at, sample FROM framework_samples ORDER BY id DESC LIMIT 1000").fetchall()
                    samples = [dict(json.loads(sample), received_at=stamp) for stamp, sample in rows]
                    body = {"source": "framework-adapter", "synthetic": False, "samples": samples}
            self.reply(200, body)

        def do_POST(self) -> None:
            validators: dict[str, Callable[[Any], dict[str, Any]]] = {
                "/api/telemetry": validate_host_sample,
                "/api/framework-metrics": validate_framework_sample,
            }
            if self.path not in validators:
                return self.reply(404, {"error": "not found"})
            if not self.authorized():
                return self.reply(401, {"error": "unauthorized"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16384:
                    return self.reply(413, {"error": "body must be 1..16384 bytes"})
                self.connection.settimeout(10)
                data = validators[self.path](json.loads(self.rfile.read(size)))
            except (ValueError, TypeError, OSError):
                return self.reply(400, {"error": "invalid sample"})
            received_at = datetime.now(timezone.utc).isoformat()
            with sqlite3.connect(database) as db:
                if self.path == "/api/telemetry":
                    db.execute("INSERT INTO samples(received_at, run_id, node, metrics) VALUES (?, ?, ?, ?)", (received_at, data["run_id"], data["node"], json.dumps(data["metrics"])))
                    db.execute("DELETE FROM samples WHERE id <= (SELECT COALESCE(MAX(id), 0) - 10000 FROM samples)")
                else:
                    db.execute("INSERT INTO framework_samples(received_at, sample) VALUES (?, ?)", (received_at, json.dumps(data)))
                    db.execute("DELETE FROM framework_samples WHERE id <= (SELECT COALESCE(MAX(id), 0) - 10000 FROM framework_samples)")
            self.reply(201, {"received_at": received_at})

    return ThreadingHTTPServer(address, Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--database", type=Path, default=Path("artifacts/observatory.sqlite3"))
    args = parser.parse_args()
    token = os.environ.get("OBSERVATORY_TOKEN", "")
    if len(token) < 24 or not token.isascii():
        parser.error("set OBSERVATORY_TOKEN to at least 24 ASCII characters")
    make_server((args.bind, args.port), args.database, token).serve_forever()


if __name__ == "__main__":
    main()
