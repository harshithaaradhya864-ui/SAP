"""AI agent that explains an alert: likely cause, severity and next step.

RuleBasedDiagnoser runs fully offline on the Raspberry Pi. It compares the
newest seconds with the healthy baseline learned during training, and with
the seconds just before the alert (to tell a sudden step from a slow drift),
then matches the pattern to known fault signatures.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .interfaces import Diagnosis, Diagnoser, Reading

ACTIONS = {
    "Loose wire or bad connection": "Switch off and isolate the supply, then check terminals, plugs and the sensor wiring for loose or burnt contacts.",
    "SMPS overload / power-supply distortion": "Reduce the load on the power supply and check its output capacitors and cooling; plan replacement if it persists.",
    "Mechanical overload or shaft friction": "Check for jamming, belt tension, shaft alignment and lubrication within 24 hours.",
    "Gradual wear (likely bearing)": "Schedule a bearing inspection (noise, heat, play) at the next planned stop; keep monitoring the trend.",
    "Under-load (load lost or belt slipping)": "Check the coupling, belt and driven equipment; the motor may be running without its load.",
    "Steady change in load level": "Check whether the machine's work changed on purpose (new load, new settings). If so, mark this as a false alarm so the model learns the new normal.",
    "Unusual current pattern": "Inspect the machine and its supply; mark this alert as a real fault or a false alarm so the model can learn.",
}


BASE_SEVERITY = {   # 0 = low, 1 = medium, 2 = high
    "Loose wire or bad connection": 2,          # fire / shock risk
    "SMPS overload / power-supply distortion": 1,
    "Mechanical overload or shaft friction": 1,
    "Under-load (load lost or belt slipping)": 1,
    "Gradual wear (likely bearing)": 0,         # days to act, but act
    "Steady change in load level": 0,
    "Unusual current pattern": 0,
}


def _mean(rs: List[Reading], attr: str) -> float:
    return float(np.mean([getattr(r, attr) or 0.0 for r in rs])) if rs else float("nan")


class RuleBasedDiagnoser(Diagnoser):
    def diagnose(self, recent: List[Reading], baseline: Dict[str, float]) -> Diagnosis:
        now = recent[-3:]                       # the abnormal seconds that raised the alert
        before = recent[-25:-5] or now          # just before the problem started
        last_min = recent[-60:]

        rms = max(r.rms for r in now)
        crest = max(r.crest_factor for r in now)
        std = max(r.std_dev for r in now)
        peak = max(r.peak for r in now)
        fused = _mean(now, "fused_score")
        if_s, ae_s = _mean(now, "if_score"), _mean(now, "ae_score")

        rms_r = rms / baseline["rms"]
        crest_r = crest / baseline["crest_factor"]
        std_r = std / max(baseline["std_dev"], 1e-6)
        jump = _mean(now, "rms") / max(_mean(before, "rms"), 1e-6)       # sudden step?
        mins = [r.rms for r in last_min]
        swing = (np.percentile(mins, 95) - np.percentile(mins, 5)) / baseline["rms"]  # rhythm over 1 min

        if std_r > 4 and rms_r < 1.08:
            cause = "Loose wire or bad connection"
            why = f"the current keeps dropping out inside each second (steadiness {std_r:.0f}x worse than healthy)"
        elif crest_r > 1.06:
            cause = "SMPS overload / power-supply distortion"
            why = (f"crest factor rose to {crest:.2f} from {baseline['crest_factor']:.2f}: "
                   "the waveform has become peaky, typical of an overloaded power supply")
        elif rms_r > 1.06 and jump > 1.04:
            cause = "Mechanical overload or shaft friction"
            why = (f"current jumped {100 * (jump - 1):.0f}% within seconds to {rms:.2f} A "
                   f"({100 * (rms_r - 1):.0f}% above healthy) while the waveform shape stayed normal")
        elif rms_r < 0.9:
            cause = "Under-load (load lost or belt slipping)"
            why = f"current fell {100 * (1 - rms_r):.0f}% below healthy"
        elif swing > 0.06 or (ae_s >= 0.5 and ae_s > if_s):
            cause = "Gradual wear (likely bearing)"
            why = (f"the current now swings by {100 * swing:.0f}% in a slow rhythm over the last minute "
                   f"and has drifted {100 * (rms_r - 1):+.0f}%; this pattern over time no longer matches healthy running")
        elif rms_r > 1.04:
            cause = "Steady change in load level"
            why = f"the current settled {100 * (rms_r - 1):.0f}% above healthy without a fault signature"
        else:
            cause = "Unusual current pattern"
            why = "the readings do not match the healthy pattern, but no known fault signature fits"

        # severity: what kind of fault it is, raised one level if it is extreme
        levels = ["low", "medium", "high"]
        level = BASE_SEVERITY[cause]
        if fused >= 0.85 or rms_r > 1.2 or crest_r > 1.2:
            level = min(level + 1, 2)
        severity = levels[level]

        explanation = (f"{cause}: {why}. Peak {peak:.2f} A; anomaly score {fused:.2f} "
                       f"(Isolation Forest {if_s:.2f}, LSTM autoencoder {ae_s:.2f}).")
        return Diagnosis(cause=cause, severity=severity, action=ACTIONS[cause], explanation=explanation)
