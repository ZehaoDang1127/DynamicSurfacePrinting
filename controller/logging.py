"""Bounded asynchronous disk logging. The motion loop never waits on disk I/O."""

from dataclasses import asdict
from pathlib import Path
import json
import queue
import threading
import time
import uuid

import numpy as np


class RunLog:
    def __init__(self, output, config, capacity=2048):
        self.path = Path(output)
        self.path.mkdir(parents=True, exist_ok=False)
        self.run_id = str(uuid.uuid4())
        self.queue = queue.Queue(maxsize=capacity)
        self.dropped = 0
        self.error = None
        self.closed = False
        self.config = config
        (self.path / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
        self.worker = threading.Thread(target=self._write, daemon=True)
        self.worker.start()

    def event(self, kind, **fields):
        self._enqueue(("event", {"kind": kind, "host_s": time.monotonic(), **fields}))

    def observation(self, observation):
        self._enqueue(("observation", observation))

    def trajectory(self, trajectory):
        self.event("proposal", **asdict(trajectory))

    def _enqueue(self, item):
        if self.closed:
            raise RuntimeError("run log is closed")
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            self.dropped += 1

    def _write(self):
        try:
            with (self.path / "events.jsonl").open("w", encoding="utf-8") as stream:
                while True:
                    item = self.queue.get()
                    try:
                        if item is None:
                            return
                        kind, value = item
                        if kind == "observation":
                            obs = value
                            filename = f"observation_{obs.frame_id:06d}.npz"
                            np.savez_compressed(self.path / filename, height_mm=obs.height_mm,
                                                valid=obs.valid, point_count=obs.point_count)
                            value = {"kind": "observation", "file": filename,
                                     **{k: v for k, v in vars(obs).items() if k not in {"height_mm", "valid", "point_count"}}}
                        stream.write(json.dumps(value, allow_nan=False) + "\n")
                    finally:
                        self.queue.task_done()
        except Exception as exc:
            self.error = str(exc)

    def finish(self, report):
        if self.closed:
            return
        # Finalization may wait on disk; the planning loop has already stopped.
        while self.worker.is_alive():
            try:
                self.queue.put(None, timeout=.1)
                break
            except queue.Full:
                continue
        self.worker.join(timeout=10)
        self.closed = True
        report = {**report, "run_id": self.run_id, "config_id": self.config.config_id,
                  "logging_drops": self.dropped, "logging_error": self.error,
                  "logging_complete": not self.worker.is_alive() and not self.error and not self.dropped}
        (self.path / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return report
