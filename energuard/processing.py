"""Reading processing: clean each one-second window and summarise it."""
from __future__ import annotations

from typing import Dict

import numpy as np

from .interfaces import FeatureSummarizer, SignalCleaner


class DespikeCleaner(SignalCleaner):
    """Removes leftover offset and isolated electrical spikes.

    mode="ac": the mean of a whole number of mains cycles is ~0, so any
               remaining mean is sensor offset and is subtracted.
    mode="dc": the Pico already removed the calibrated zero offset; the mean
               IS the load current, so it is kept.
    Spikes: a single sample that jumps far away from both of its neighbours
    (electrical interference) is replaced by the median of its neighbours.
    """

    def __init__(self, mode: str = "ac", k: float = 8.0):
        if mode not in ("ac", "dc"):
            raise ValueError("mode must be 'ac' or 'dc'")
        self.mode = mode
        self.k = k

    def clean(self, samples: np.ndarray) -> np.ndarray:
        x = np.asarray(samples, dtype=float).copy()
        # 1) spikes: a sample far from its neighbours.
        #    Neighbours = median of the 2 samples on each side. The limit is
        #    at least half the signal amplitude, so real waveform
        #    peaks (even sharp SMPS peaks) are never touched.
        pad = np.pad(x, 2, mode="edge")
        nb = np.median(np.stack([pad[:-4], pad[1:-3], pad[3:-1], pad[4:]]), axis=0)   # 4 neighbours
        resid = x - nb
        mad = np.median(np.abs(resid - np.median(resid))) * 1.4826
        amp = np.percentile(np.abs(x - np.median(x)), 99)
        spikes = np.abs(resid) > max(self.k * mad, 0.5 * amp, 0.02)
        x[spikes] = nb[spikes]
        # 2) offset
        if self.mode == "ac":
            x -= x.mean()
        return x


class CurrentSummarizer(FeatureSummarizer):
    """rms, peak, std_dev and crest_factor for one window.

    For AC, std_dev is the variation of the RMS from one mains cycle to the
    next (how steady the current is); for DC it is the plain std deviation.
    """

    def __init__(self, mode: str = "ac", fs: int = 860, mains_hz: float = 50.0):
        self.mode = mode
        self.cycle = max(2, int(round(fs / mains_hz)))

    def summarize(self, x: np.ndarray) -> Dict[str, float]:
        rms = float(np.sqrt(np.mean(x ** 2)))
        peak = float(np.max(np.abs(x)))
        if self.mode == "ac":
            n = (x.size // self.cycle) * self.cycle
            cyc = np.sqrt(np.mean(x[:n].reshape(-1, self.cycle) ** 2, axis=1))
            std = float(np.std(cyc))
        else:
            std = float(np.std(x))
        crest = peak / rms if rms > 1e-6 else 0.0
        return {"rms": rms, "peak": peak, "std_dev": std, "crest_factor": crest}
