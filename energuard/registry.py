"""ModelRegistry: keeps every trained model version and marks one as active.

models/
  registry.json          list of versions + which one is active
  v1/  meta.json, isolation_forest.joblib, lstm_autoencoder.pt/.json
  v2/  ...
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

from .detectors import DETECTOR_TYPES
from .interfaces import ModelBundle, ModelRegistry


class FileModelRegistry(ModelRegistry):
    def __init__(self, root: str = "models"):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.index_path = os.path.join(root, "registry.json")
        if not os.path.exists(self.index_path):
            self._write_index({"active": None, "versions": []})

    def _index(self) -> dict:
        with open(self.index_path) as f:
            return json.load(f)

    def _write_index(self, idx: dict) -> None:
        tmp = self.index_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(idx, f, indent=2)
        os.replace(tmp, self.index_path)      # atomic: the live pipeline never reads a half-written file

    def register(self, bundle: ModelBundle, metrics: Dict[str, float]) -> int:
        idx = self._index()
        version = max([v["version"] for v in idx["versions"]], default=0) + 1
        folder = os.path.join(self.root, f"v{version}")
        os.makedirs(folder, exist_ok=True)
        for d in bundle.detectors:
            d.save(folder)
        meta = {"version": version, "trained_at": time.time(),
                "detectors": [d.name for d in bundle.detectors],
                "baseline": bundle.baseline, "metrics": metrics, **bundle.meta}
        with open(os.path.join(folder, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        idx["versions"].append({"version": version, "trained_at": meta["trained_at"], "metrics": metrics,
                                "note": bundle.meta.get("note", "")})
        self._write_index(idx)
        bundle.version = version
        return version

    def activate(self, version: int) -> None:
        idx = self._index()
        if version not in [v["version"] for v in idx["versions"]]:
            raise ValueError(f"unknown model version {version}")
        idx["active"] = version
        self._write_index(idx)

    def active_version(self) -> Optional[int]:
        return self._index()["active"]

    def load(self, version: Optional[int] = None) -> ModelBundle:
        version = version or self.active_version()
        if version is None:
            raise RuntimeError("No trained model yet. Run `python -m energuard train` first.")
        folder = os.path.join(self.root, f"v{version}")
        with open(os.path.join(folder, "meta.json")) as f:
            meta = json.load(f)
        detectors = []
        for name in meta["detectors"]:
            d = DETECTOR_TYPES[name]()
            d.load(folder)
            detectors.append(d)
        return ModelBundle(version=version, detectors=detectors, baseline=meta["baseline"], meta=meta)

    def versions(self) -> List[dict]:
        idx = self._index()
        return [dict(v, active=(v["version"] == idx["active"])) for v in idx["versions"]]
