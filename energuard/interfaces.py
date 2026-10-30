"""Abstract interfaces for every EnerGuard building block.

The pipeline only ever talks to these abstractions (Dependency Inversion).
Each interface is small and has one job (Single Responsibility, Interface
Segregation). New behaviour is added as a new implementation class instead
of editing the pipeline (Open/Closed), and any implementation can replace
another as long as it keeps the contract described here (Liskov).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

FEATURES = ["rms", "peak", "std_dev", "crest_factor"]


# --------------------------------------------------------------------------- data
@dataclass
class Window:
    """One second of current samples (amps) from one machine."""
    device_id: str
    ts: float                      # epoch seconds at the start of the window
    samples: np.ndarray            # ~860 current values in amps
    truth: Optional[str] = None    # simulator only: name of the active fault, if any


@dataclass
class Reading:
    """One stored row per second (raw_1s)."""
    device_id: str
    load_type: str
    ts: float
    rms: float
    peak: float
    std_dev: float
    crest_factor: float
    if_score: Optional[float] = None
    ae_score: Optional[float] = None
    fused_score: Optional[float] = None
    model_version: Optional[int] = None
    truth: Optional[str] = None

    def features(self) -> np.ndarray:
        return np.array([self.rms, self.peak, self.std_dev, self.crest_factor], dtype=float)


@dataclass
class Diagnosis:
    cause: str
    severity: str          # low / medium / high
    action: str
    explanation: str


@dataclass
class Alert:
    alert_id: str
    device_id: str
    ts: float
    fused_score: float
    duration_s: int
    severity: str
    cause: str
    action: str
    diagnosis: str
    model_version: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Feedback:
    alert_id: str
    device_id: str
    label: str             # "fault" or "false_alarm"
    ts: float
    note: str = ""


@dataclass
class ModelBundle:
    """Everything the live pipeline needs from one trained model version."""
    version: int
    detectors: List["AnomalyDetector"]
    baseline: Dict[str, float]                  # healthy mean of each feature
    meta: Dict[str, object] = field(default_factory=dict)


# --------------------------------------------------------------------------- interfaces
class SensorSource(ABC):
    """Produces one-second windows of current samples."""

    @abstractmethod
    def windows(self) -> Iterator[Window]:
        ...


class SignalCleaner(ABC):
    """Removes offset and noise from one window of samples."""

    @abstractmethod
    def clean(self, samples: np.ndarray) -> np.ndarray:
        ...


class FeatureSummarizer(ABC):
    """Turns one cleaned window into a small set of summary values."""

    @abstractmethod
    def summarize(self, samples: np.ndarray) -> Dict[str, float]:
        ...


class AnomalyDetector(ABC):
    """Learns normal behaviour and scores new data on a 0-1 scale.

    Contract every detector must keep (so any one can replace another):
      * fit() uses healthy data only.
      * score() returns 0 for typical healthy data, 0.5 at the edge of
        healthy behaviour (99th percentile), and is capped at 1.0.
    """
    name: str = "detector"

    @abstractmethod
    def fit(self, X: np.ndarray, X_calib: np.ndarray) -> None:
        """X, X_calib: healthy feature rows, shape (n_seconds, n_features)."""

    @abstractmethod
    def score(self, x: np.ndarray) -> float:
        """Score the newest feature row (detectors may keep their own history)."""

    def score_series(self, X: np.ndarray) -> np.ndarray:
        """Score many consecutive rows at once (offline replay). Same result
        as calling score() row by row from a fresh start."""
        self.reset()
        out = np.array([self.score(x) for x in X])
        self.reset()
        return out

    @abstractmethod
    def reset(self) -> None:
        """Forget any internal history (e.g. at the start of a new run)."""

    @abstractmethod
    def save(self, folder: str) -> None:
        ...

    @abstractmethod
    def load(self, folder: str) -> None:
        ...


class ScoreFusion(ABC):
    @abstractmethod
    def fuse(self, scores: Dict[str, float]) -> float:
        ...


class AlertPolicy(ABC):
    """Decides when a stream of scores becomes an alert."""

    @abstractmethod
    def update(self, score: float, ts: float) -> Optional[int]:
        """Return the number of abnormal seconds when a new alert starts, else None."""

    @abstractmethod
    def reset(self) -> None:
        ...


class Diagnoser(ABC):
    @abstractmethod
    def diagnose(self, recent: List[Reading], baseline: Dict[str, float]) -> Diagnosis:
        ...


class Notifier(ABC):
    @abstractmethod
    def send(self, alert: Alert) -> None:
        ...


class ReadingStore(ABC):
    @abstractmethod
    def write_reading(self, r: Reading) -> None: ...

    @abstractmethod
    def write_alert(self, a: Alert) -> None: ...

    @abstractmethod
    def write_feedback(self, f: Feedback) -> None: ...

    @abstractmethod
    def readings(self, device_id: Optional[str] = None, start: Optional[float] = None,
                 end: Optional[float] = None) -> pd.DataFrame: ...

    @abstractmethod
    def alerts(self, device_id: Optional[str] = None) -> pd.DataFrame: ...

    @abstractmethod
    def feedback(self) -> pd.DataFrame: ...

    @abstractmethod
    def downsample(self) -> int:
        """Aggregate raw_1s into agg_1m. Returns rows written."""

    @abstractmethod
    def apply_retention(self, now: float) -> None: ...

    def flush(self) -> None:
        """Write any buffered rows (optional)."""


class ModelRegistry(ABC):
    @abstractmethod
    def register(self, bundle: ModelBundle, metrics: Dict[str, float]) -> int: ...

    @abstractmethod
    def activate(self, version: int) -> None: ...

    @abstractmethod
    def active_version(self) -> Optional[int]: ...

    @abstractmethod
    def load(self, version: Optional[int] = None) -> ModelBundle: ...

    @abstractmethod
    def versions(self) -> List[dict]: ...
