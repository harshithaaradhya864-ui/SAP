"""Score fusion and the alert rule."""
from __future__ import annotations

from typing import Dict, Optional

from .interfaces import AlertPolicy, ScoreFusion


class WeightedFusion(ScoreFusion):
    """Combine the detectors' 0-1 scores into one.

    fused = max( weighted average , strong * highest single score )

    * The weighted average needs both models to lean the same way, which
      keeps false alarms low.
    * The second term lets ONE model raise the alarm on its own when it is
      very sure (with strong = 0.8, a single score above 0.625 is enough).
      Without it, a steady fault that only the Isolation Forest sees (the
      autoencoder calms down once the level stops changing) would sit at
      exactly 0.5 and never alert.
    Weights are re-normalised over the detectors present, so the system still
    works if one detector is switched off.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None, strong: float = 0.8):
        self.weights = weights or {"if": 0.5, "ae": 0.5}
        self.strong = strong

    def fuse(self, scores: Dict[str, float]) -> float:
        used = {k: v for k, v in scores.items() if v is not None and self.weights.get(k, 0) > 0}
        if not used:
            return 0.0
        total = sum(self.weights[k] for k in used)
        average = sum(self.weights[k] * v for k, v in used.items()) / total
        return max(average, self.strong * max(used.values()))


class ConsecutiveAlertPolicy(AlertPolicy):
    """Alert when the score stays above `threshold` for `consecutive` seconds.

    After an alert, the same problem does not raise new alerts until the
    score has been back below the threshold for `clear_after` seconds.
    """

    def __init__(self, threshold: float = 0.5, consecutive: int = 3, clear_after: int = 60):
        self.threshold = threshold
        self.consecutive = consecutive
        self.clear_after = clear_after
        self.reset()

    def reset(self):
        self._high = 0
        self._low = 0
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def update(self, score: float, ts: float) -> Optional[int]:
        if score > self.threshold:
            self._high += 1
            self._low = 0
        else:
            self._high = 0
            self._low += 1
            if self._active and self._low >= self.clear_after:
                self._active = False
        if not self._active and self._high >= self.consecutive:
            self._active = True
            return self._high
        return None
