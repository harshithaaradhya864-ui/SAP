"""The live monitoring loop. It depends only on interfaces (Dependency Inversion)."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

from .interfaces import (Alert, AlertPolicy, Diagnoser, FeatureSummarizer, FEATURES, ModelBundle,
                         ModelRegistry, Notifier, Reading, ReadingStore, ScoreFusion, SensorSource,
                         SignalCleaner, Window)


@dataclass
class RunStats:
    seconds: int = 0
    alerts: List[Alert] = field(default_factory=list)
    model_versions: set = field(default_factory=set)


class Pipeline:
    def __init__(self, cleaner: SignalCleaner, summarizer: FeatureSummarizer, store: ReadingStore,
                 load_type: str = "motor", registry: Optional[ModelRegistry] = None,
                 fusion: Optional[ScoreFusion] = None, policy: Optional[AlertPolicy] = None,
                 diagnoser: Optional[Diagnoser] = None, notifier: Optional[Notifier] = None,
                 reload_every: int = 60, maintenance_every: int = 60):
        self.cleaner, self.summarizer, self.store = cleaner, summarizer, store
        self.load_type = load_type
        self.registry, self.fusion, self.policy = registry, fusion, policy
        self.diagnoser, self.notifier = diagnoser, notifier
        self.reload_every, self.maintenance_every = reload_every, maintenance_every
        self.bundle: Optional[ModelBundle] = None
        self.recent: deque = deque(maxlen=120)
        self._n = 0
        self._last_alerts: List[Alert] = []
        if registry is not None:
            self._load_active_model()

    # ------------------------------------------------------------------ model
    def _load_active_model(self) -> None:
        v = self.registry.active_version()
        if v is not None and (self.bundle is None or self.bundle.version != v):
            self.bundle = self.registry.load(v)
            print(f"[pipeline] using model version v{v}")

    @property
    def monitoring(self) -> bool:
        return self.bundle is not None

    # ------------------------------------------------------------------ one second
    def process(self, w: Window) -> Reading:
        clean = self.cleaner.clean(w.samples)
        f = self.summarizer.summarize(clean)
        r = Reading(device_id=w.device_id, load_type=self.load_type, ts=w.ts, truth=w.truth, **f)

        if self.monitoring:
            x = r.features()
            scores = {d.name: d.score(x) for d in self.bundle.detectors}
            r.if_score, r.ae_score = scores.get("if"), scores.get("ae")
            r.fused_score = self.fusion.fuse(scores)
            r.model_version = self.bundle.version

        self.store.write_reading(r)
        self.recent.append(r)

        if self.monitoring:
            n_high = self.policy.update(r.fused_score, r.ts)
            if n_high is not None:
                self._raise_alert(r, n_high)

        self._n += 1
        if self.registry is not None and self._n % self.reload_every == 0:
            self._load_active_model()                 # hot-swap to a newly deployed version
        if self._n % self.maintenance_every == 0:
            self.store.downsample()
        return r

    def _raise_alert(self, r: Reading, n_high: int) -> Alert:
        d = self.diagnoser.diagnose(list(self.recent), self.bundle.baseline)
        a = Alert(alert_id=f"{r.device_id}-{int(r.ts)}", device_id=r.device_id, ts=r.ts,
                  fused_score=round(r.fused_score, 3), duration_s=n_high, severity=d.severity,
                  cause=d.cause, action=d.action, diagnosis=d.explanation, model_version=self.bundle.version)
        self.store.write_alert(a)
        if self.notifier:
            self.notifier.send(a)
        self._last_alerts.append(a)
        return a

    # ------------------------------------------------------------------ loop
    def run(self, source: SensorSource, max_seconds: Optional[int] = None, progress_every: int = 0) -> RunStats:
        stats = RunStats()
        self._last_alerts = stats.alerts
        t0 = time.time()
        for w in source.windows():
            r = self.process(w)
            stats.seconds += 1
            if r.model_version:
                stats.model_versions.add(r.model_version)
            if progress_every and stats.seconds % progress_every == 0:
                score = f"score {r.fused_score:.2f}" if r.fused_score is not None else "recording"
                print(f"  {stats.seconds:>6d} s  rms {r.rms:.3f} A  {score}  alerts {len(stats.alerts)}"
                      f"  ({time.time() - t0:.0f}s elapsed)")
            if max_seconds and stats.seconds >= max_seconds:
                break
        self.store.flush()
        self.store.downsample()
        return stats
