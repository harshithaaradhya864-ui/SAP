"""Trainer: builds model versions from healthy data and decides whether a
retrained version is good enough to deploy."""
from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .interfaces import AlertPolicy, FEATURES, ModelBundle, ReadingStore, ScoreFusion


class Trainer:
    def __init__(self, detector_factories: Dict[str, Callable], calib_fraction: float = 0.2):
        self.factories = detector_factories
        self.calib_fraction = calib_fraction

    def train(self, healthy: pd.DataFrame, note: str = "") -> ModelBundle:
        X = healthy[FEATURES].to_numpy(dtype=float)
        if len(X) < 600:
            raise ValueError(f"Need at least 10 minutes of healthy data, got {len(X)} s")
        split = int(len(X) * (1 - self.calib_fraction))
        X_fit, X_cal = X[:split], X[split:]          # calibrate the 0-1 scale on data the model has not seen
        detectors = []
        for name, make in self.factories.items():
            d = make()
            d.fit(X_fit, X_cal)
            detectors.append(d)
        baseline = {k: float(healthy[k].mean()) for k in FEATURES}
        return ModelBundle(version=0, detectors=detectors, baseline=baseline,
                           meta={"note": note, "healthy_seconds": int(len(X))})


# ------------------------------------------------------------------------- replay
def replay(bundle: ModelBundle, X: np.ndarray, fusion: ScoreFusion, policy: AlertPolicy,
           detector_names: Optional[List[str]] = None) -> Tuple[np.ndarray, List[int]]:
    """Run stored feature rows through a model offline.
    Returns (fused score per second, indices where alerts start)."""
    policy = copy.deepcopy(policy)
    policy.reset()
    dets = [d for d in bundle.detectors if detector_names is None or d.name in detector_names]
    series = {d.name: d.score_series(X) for d in dets}
    fused, starts = np.zeros(len(X)), []
    for i in range(len(X)):
        fused[i] = fusion.fuse({k: v[i] for k, v in series.items()})
        if policy.update(fused[i], float(i)) is not None:
            starts.append(i)
    return fused, starts


def fault_periods(readings: pd.DataFrame, alerts: pd.DataFrame, feedback: pd.DataFrame,
                  label: str, before: float = 0, clear_after: int = 60) -> List[Tuple[float, float]]:
    """Time ranges around alerts the user labelled `label`.
    A period runs from the alert (minus `before` seconds) until the stored
    score has stayed below 0.5 for `clear_after` seconds."""
    ids = set(feedback.loc[feedback["label"] == label, "alert_id"])
    out = []
    ts = readings["ts"].to_numpy()
    low = (readings["fused_score"].fillna(0) <= 0.5).to_numpy()
    for _, a in alerts[alerts["alert_id"].isin(ids)].iterrows():
        i = int(np.searchsorted(ts, a["ts"]))
        run, end = 0, ts[-1] if len(ts) else a["ts"]
        for j in range(i, len(ts)):
            run = run + 1 if low[j] else 0
            if run >= clear_after:
                end = ts[j]
                break
        out.append((a["ts"] - before, end))
    return out


def _mask(ts: np.ndarray, periods) -> np.ndarray:
    m = np.zeros(len(ts), dtype=bool)
    for a, b in periods:
        m |= (ts >= a) & (ts <= b)
    return m


def build_retraining_data(store: ReadingStore, original_healthy: pd.DataFrame,
                          device_id: str, clearly_normal: float = 0.3) -> Dict[str, object]:
    """Collect trusted healthy data and labelled test data from the store.

    Trusted healthy = the original healthy recording
                    + live seconds the current model scored CLEARLY normal (<= 0.3)
                      that are far (10 min) from any alert not confirmed as a false alarm
                    + every period the user marked as FALSE ALARM (now known to be
                      normal); its onset is kept back to test the new model
    Confirmed faults are never trained on; they are kept to test the new model.
    """
    live = store.readings(device_id=device_id)
    live = live[live["model_version"].notna()].reset_index(drop=True)
    alerts, fb = store.alerts(device_id), store.feedback()
    false_ids = set(fb.loc[fb["label"] == "false_alarm", "alert_id"]) if len(fb) else set()
    faults = fault_periods(live, alerts, fb, "fault", before=600)
    false_alarms = fault_periods(live, alerts, fb, "false_alarm")
    ts = live["ts"].to_numpy()
    risky = [(a - 600, a + 600) for a, i in zip(alerts["ts"], alerts["alert_id"]) if i not in false_ids]
    near_alert = _mask(ts, risky) | _mask(ts, [(a, b + 300) for a, b in faults])
    in_false = _mask(ts, false_alarms)
    clearly_ok = (live["fused_score"] <= clearly_normal).to_numpy()

    # automatically trusted seconds: first 80 % to train, newest 20 % to test
    auto = live[clearly_ok & ~near_alert & ~in_false]
    split = auto["ts"].quantile(0.8) if len(auto) else 0
    parts_train = [original_healthy, auto[auto["ts"] < split]]
    healthy_test = auto[auto["ts"] >= split]

    # user-confirmed false alarms: test on the onset (2 min before the alert
    # to 30 % into the period), learn the settled "new normal" from the rest
    false_alarm_segments = []
    for a, b in false_alarms:
        cut = a + 0.3 * (b - a)
        false_alarm_segments.append(live[(live["ts"] >= a - 120) & (live["ts"] < cut)])
        parts_train.append(live[(live["ts"] >= cut) & (live["ts"] <= b)])

    healthy_train = pd.concat(parts_train).sort_values("ts")[FEATURES + ["ts"]]
    fault_segments = [live[(live["ts"] >= a) & (live["ts"] <= b)] for a, b in faults]
    return {"healthy_train": healthy_train, "healthy_test": healthy_test,
            "fault_segments": fault_segments, "false_alarm_segments": false_alarm_segments,
            "n_fault": len(faults), "n_false": len(false_alarms)}


def compare(bundle: ModelBundle, data: Dict[str, object], fusion: ScoreFusion,
            policy: AlertPolicy) -> Dict[str, float]:
    """Score a model on labelled data: false alarms (lower is better) and
    confirmed faults caught (higher is better)."""
    fa = 0
    hours = max(len(data["healthy_test"]) / 3600, 1e-9)
    if len(data["healthy_test"]):
        _, s = replay(bundle, data["healthy_test"][FEATURES].to_numpy(), fusion, policy)
        fa += len(s)
    for seg in data["false_alarm_segments"]:
        if len(seg):
            fa += len(replay(bundle, seg[FEATURES].to_numpy(), fusion, policy)[1])
    caught = 0
    for seg in data["fault_segments"]:
        if len(seg) and replay(bundle, seg[FEATURES].to_numpy(), fusion, policy)[1]:
            caught += 1
    return {"false_alarms": fa, "false_alarm_periods": len(data["false_alarm_segments"]),
            "healthy_test_minutes": round(hours * 60, 1),
            "faults_caught": caught, "faults_total": len(data["fault_segments"])}
