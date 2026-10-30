"""Measure detection quality on simulated data (where the true faults are known)."""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from .interfaces import FEATURES, ModelBundle
from .trainer import replay

GRACE = 30  # an alert up to 30 s after a fault ends still counts as detecting it


def fault_episodes(readings: pd.DataFrame) -> List[dict]:
    truth = readings["truth"].fillna("").to_numpy()
    ts = readings["ts"].to_numpy()
    eps, i = [], 0
    while i < len(truth):
        if truth[i]:
            j = i
            while j + 1 < len(truth) and truth[j + 1] == truth[i]:
                j += 1
            eps.append({"fault": truth[i], "start": ts[i], "end": ts[j]})
            i = j + 1
        else:
            i += 1
    return eps


def score_alerts(readings: pd.DataFrame, alert_ts: List[float]) -> Dict[str, object]:
    eps = fault_episodes(readings)
    used = set()
    rows = []
    for e in eps:
        hit = [t for t in alert_ts if e["start"] <= t <= e["end"] + GRACE]
        if hit:
            used.update(hit)
        rows.append({"fault": e["fault"], "duration_s": int(e["end"] - e["start"] + 1),
                     "detected": bool(hit), "delay_s": int(min(hit) - e["start"]) if hit else None})
    false = [t for t in alert_ts if t not in used]
    healthy_h = float((readings["truth"].isna()).sum()) / 3600
    return {"faults": rows, "faults_detected": sum(r["detected"] for r in rows), "faults_total": len(rows),
            "false_alarms": len(false), "false_alarm_ts": false,
            "healthy_hours": round(healthy_h, 2),
            "false_alarms_per_hour": round(len(false) / max(healthy_h, 1e-9), 2)}


def ablation(bundle: ModelBundle, readings: pd.DataFrame, fusion, policy) -> Dict[str, dict]:
    """Same data, three set-ups: Isolation Forest only, autoencoder only, both."""
    X = readings[FEATURES].to_numpy()
    ts = readings["ts"].to_numpy()
    names = [d.name for d in bundle.detectors]
    setups = {"Isolation Forest only": ["if"], "LSTM autoencoder only": ["ae"], "Both (fused)": names}
    out = {}
    for label, dets in setups.items():
        if not set(dets) <= set(names):
            continue
        _, starts = replay(bundle, X, fusion, policy, dets)
        out[label] = score_alerts(readings, [ts[i] for i in starts])
    return out


def plot(readings: pd.DataFrame, alerts: pd.DataFrame, path: str, threshold: float = 0.5,
         harmless: list = None) -> None:
    """harmless: [(start_s, label)] events that are NOT faults, drawn as grey markers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = (readings["ts"] - readings["ts"].iloc[0]) / 60
    t0 = readings["ts"].iloc[0]
    fig, ax = plt.subplots(3, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [1.2, 1, 1]})
    navy, red, amber = "#213163", "#B5443B", "#F2C14E"
    for e in fault_episodes(readings):
        for a in ax:
            a.axvspan((e["start"] - t0) / 60, (e["end"] - t0) / 60, color=amber, alpha=0.25, lw=0)
    ax[0].plot(t, readings["rms"], color=navy, lw=0.8)
    ax[0].set_ylabel("RMS current (A)")
    ax[1].plot(t, readings["if_score"], color="#185FA5", lw=0.7, label="Isolation Forest (ML)")
    if readings["ae_score"].notna().any():
        ax[1].plot(t, readings["ae_score"], color="#0F6E56", lw=0.7, label="LSTM autoencoder (DL)")
    ax[1].set_ylabel("Model score")
    ax[1].legend(loc="upper left", frameon=False, fontsize=9)
    ax[2].plot(t, readings["fused_score"], color=navy, lw=0.7)
    ax[2].axhline(threshold, color=red, ls="--", lw=1)
    ax[2].text(0.3, threshold + 0.03, "threshold", color=red, fontsize=9)
    for _, a in alerts.iterrows():
        ax[2].axvline((a["ts"] - t0) / 60, color=red, lw=1)
    ax[2].set_ylabel("Fused score")
    ax[2].set_xlabel("Time (minutes)")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    ymax = readings["rms"].max() * 1.02
    for e in fault_episodes(readings):
        ax[0].text((e["start"] - t0) / 60 + 0.3, ymax, e["fault"].replace("_", " "), fontsize=9,
                   va="top", color="#633806")
    for start, lab in harmless or []:
        x = start / 60
        for a in ax:
            a.axvline(x, color="#8A8780", ls=":", lw=1)
        ax[0].text(x + 0.3, readings["rms"].min(), lab, fontsize=8, color="#5F5E5A", va="bottom")
    ax[1].set_ylim(0, 1.05)
    ax[2].set_ylim(0, 1.05)
    fig.suptitle("EnerGuard on the simulated test run: shaded = injected fault, red lines = alerts, grey dotted = harmless events",
                 fontsize=11, color=navy, x=0.07, ha="left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_report(report: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        raise TypeError(type(o))

    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=conv)
