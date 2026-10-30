"""Realistic motor / SMPS current simulator with fault injection.

Produces the same thing the real hardware does: about 860 current samples
per second (amps), so the rest of EnerGuard cannot tell the difference.

Model: 50 Hz AC load current with small motor harmonics, slow natural load
fluctuation, ACS712 sensor noise and ADS1115 quantisation.
Faults can be scheduled at any time:
  friction     - mechanical overload / shaft friction (current rises ~25 %)
  smps         - SMPS overload: strong 3rd/5th harmonics, peaky waveform
  loose_wire   - intermittent contact: short drop-outs in the current
  bearing      - slow bearing wear: gradual rise plus a growing 12 s ripple
  startup      - normal motor start-up inrush (NOT a fault; tests false alarms)
  load_change  - the machine is given a permanently heavier load (NOT a fault:
                 a "new normal" that the model must learn through feedback)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

FS = 860            # samples per second (ADS1115 maximum rate)
MAINS_HZ = 50.0
FAULTS = ("friction", "smps", "loose_wire", "bearing")
NOT_FAULTS = ("startup", "load_change")


@dataclass
class FaultEvent:
    kind: str
    start: float        # seconds from the beginning of the simulation
    duration: float

    def active(self, t: float) -> bool:
        return self.start <= t < self.start + self.duration

    def progress(self, t: float) -> float:
        return float(np.clip((t - self.start) / max(self.duration, 1e-9), 0.0, 1.0))


class MotorCurrentSimulator:
    def __init__(self, rated_rms: float = 0.8, seed: int = 0, fs: int = FS,
                 noise_a: float = 0.012, events: Optional[List[FaultEvent]] = None,
                 harmonic_phases: tuple = (0.6, 1.9), fluctuation: float = 0.0025):
        self.rated_rms = rated_rms
        self.fluctuation = fluctuation          # natural load wander (~1 % typical)
        self.fs = fs
        self.noise_a = noise_a
        self.rng = np.random.default_rng(seed)     # seed changes the noise, not the machine
        self.events = list(events or [])
        self._fluct = 0.0
        # the waveform shape belongs to the machine, so it is the same on every run
        self._phase3, self._phase5 = harmonic_phases

    # ------------------------------------------------------------------ helpers
    def active_fault(self, t: float) -> Optional[str]:
        for e in self.events:
            if e.kind not in NOT_FAULTS and e.active(t):
                return e.kind
        return None

    def _ramp(self, e: FaultEvent, t: np.ndarray, rise: float = 5.0) -> np.ndarray:
        return np.clip((t - e.start) / rise, 0.0, 1.0) * ((t >= e.start) & (t < e.start + e.duration))

    # ------------------------------------------------------------------ main
    def second(self, sec: int) -> np.ndarray:
        """Return the FS current samples for simulation second `sec`."""
        t = sec + np.arange(self.fs) / self.fs
        w = 2 * np.pi * MAINS_HZ

        # slow natural load fluctuation (bounded random walk)
        self._fluct = 0.97 * self._fluct + self.rng.normal(0, self.fluctuation)
        amp = np.full_like(t, np.sqrt(2) * self.rated_rms * (1 + self._fluct))
        h3 = np.full_like(t, 0.08)
        h5 = np.full_like(t, 0.04)
        dropout = np.ones_like(t)

        for e in self.events:
            if not (e.start - 1 <= sec < e.start + e.duration + 1):
                continue
            r = self._ramp(e, t)
            if e.kind == "friction":
                amp *= 1 + 0.25 * r
            elif e.kind == "smps":
                h3 += 0.40 * r
                h5 += 0.22 * r
                amp *= 1 + 0.08 * r
            elif e.kind == "bearing":
                p = np.clip((t - e.start) / e.duration, 0, 1) * (t >= e.start) * (t < e.start + e.duration)
                amp *= 1 + 0.07 * p + 0.09 * p * np.sin(2 * np.pi * t / 12.0)
            elif e.kind == "loose_wire" and e.active(sec):
                for _ in range(self.rng.integers(1, 4)):
                    if self.rng.random() < 0.8:
                        a = self.rng.integers(0, self.fs - 110)
                        n = self.rng.integers(25, 110)          # 30-130 ms gap
                        dropout[a:a + n] = 0.05
                amp *= 1 + self.rng.normal(0, 0.03)
            elif e.kind == "load_change":
                amp *= 1 + 0.07 * r
            elif e.kind == "startup":
                tau = t - e.start
                inrush = np.where((tau >= 0) & (tau < 1.0), 4.0 * np.exp(-np.maximum(tau, 0) / 0.15), 0.0)
                amp *= 1 + inrush

        i = amp * (np.sin(w * t) + h3 * np.sin(3 * w * t + self._phase3) + h5 * np.sin(5 * w * t + self._phase5))
        i *= dropout
        i += self.rng.normal(0, self.noise_a, size=t.size)            # ACS712 noise
        # ADS1115 quantisation: 0.125 mV/LSB at +-4.096 V, through 2/3 divider and 185 mV/A
        lsb_amps = 0.000125 / (2 / 3) / 0.185
        return np.round(i / lsb_amps) * lsb_amps


def evaluation_scenario() -> List[FaultEvent]:
    """A 110-minute test timeline: four different faults, three harmless
    motor start-ups, long healthy stretches to count false alarms, and at the
    end a harmless permanent load increase (a "new normal")."""
    return [
        FaultEvent("friction", 600, 120),
        FaultEvent("smps", 1200, 120),
        FaultEvent("loose_wire", 1800, 60),
        FaultEvent("bearing", 2400, 1800),
        FaultEvent("startup", 4500, 1),
        FaultEvent("startup", 5000, 1),
        FaultEvent("startup", 5500, 1),
        FaultEvent("load_change", 5900, 700),
    ]
