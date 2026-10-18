"""Upload trigger: watch a landing directory and run the pipeline once per new file.

Locally this stands in for an S3 ObjectCreated notification. A file is only picked up
once its size is stable, and each file is processed exactly once (tracked by name+hash),
so a crash mid-run never retrains twice on the same upload.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

from .pipeline import Pipeline

logger = logging.getLogger("mlops.trigger")


class LandingWatcher:
    def __init__(self, landing: Path, pipeline: Pipeline):
        self.landing = landing
        self.pipeline = pipeline
        self.seen_path = pipeline.state / "trigger_seen.json"
        self.seen: dict[str, str] = json.loads(self.seen_path.read_text()) if self.seen_path.exists() else {}
        self._sizes: dict[str, int] = {}

    def _stable(self, p: Path) -> bool:
        size = p.stat().st_size
        stable = self._sizes.get(p.name) == size
        self._sizes[p.name] = size
        return stable

    def poll_once(self) -> list:
        results = []
        for p in sorted(self.landing.glob("*.csv")):
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            if self.seen.get(p.name) == digest or not self._stable(p):
                continue
            results.append(self.pipeline.process(p))
            self.seen[p.name] = digest
            self.seen_path.write_text(json.dumps(self.seen, indent=2))
        return results

    def run_forever(self, interval: float = 2.0) -> None:
        logger.info("watching %s every %.1fs", self.landing, interval)
        while True:
            self.poll_once()
            time.sleep(interval)
