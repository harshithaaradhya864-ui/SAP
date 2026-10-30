"""Composition root: the only place that knows which concrete classes are used.

Everything else receives its collaborators through the constructor, so a
different database, sensor or alert channel is chosen here from config.yaml.
"""
from __future__ import annotations

from typing import Dict, Optional

import yaml

from .decision import ConsecutiveAlertPolicy, WeightedFusion
from .detectors import IsolationForestDetector, LSTMAutoencoderDetector, torch_available
from .diagnoser import RuleBasedDiagnoser
from .notifiers import ConsoleNotifier, EmailNotifier, MultiNotifier, TelegramNotifier
from .pipeline import Pipeline
from .processing import CurrentSummarizer, DespikeCleaner
from .registry import FileModelRegistry
from .simulator import FaultEvent, MotorCurrentSimulator
from .sources import SerialSource, SimulatedSource
from .storage import InfluxStore, SQLiteStore
from .trainer import Trainer


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_store(cfg):
    s = cfg["store"]
    if s["type"] == "influx":
        return InfluxStore(**s["influx"])
    return SQLiteStore(s["sqlite_path"])


def make_registry(cfg):
    return FileModelRegistry(cfg["models"]["root"])


def make_fusion(cfg):
    return WeightedFusion(cfg["models"]["fusion_weights"], cfg["models"].get("fusion_strong", 0.8))


def make_policy(cfg):
    a = cfg["alerts"]
    return ConsecutiveAlertPolicy(a["threshold"], a["consecutive"], a["clear_after"])


def make_trainer(cfg) -> Trainer:
    m = cfg["models"]
    factories = {}
    if m.get("isolation_forest", True):
        factories["if"] = IsolationForestDetector
    if m.get("lstm_autoencoder", True):
        if torch_available():
            ae = m.get("ae", {})
            factories["ae"] = lambda: LSTMAutoencoderDetector(seq_len=ae.get("seq_len", 60),
                                                              epochs=ae.get("epochs", 30))
        else:
            print("[factory] PyTorch not installed: training Isolation Forest only "
                  "(pip install torch to enable the LSTM autoencoder)")
    return Trainer(factories)


def make_notifier(cfg, quiet: bool = False):
    n = cfg["notify"]
    chans = []
    if n.get("console", True):
        chans.append(ConsoleNotifier(quiet=quiet))
    if n["telegram"]["enabled"]:
        chans.append(TelegramNotifier(n["telegram"]["bot_token"], str(n["telegram"]["chat_id"])))
    if n["email"]["enabled"]:
        e = n["email"]
        chans.append(EmailNotifier(e["smtp_host"], e["smtp_port"], e["user"], e["password"], e["to"]))
    return MultiNotifier(chans)


def make_source(cfg, seconds: int, events=None, seed: int = 0, start_ts: Optional[float] = None,
                realtime: Optional[bool] = None):
    s = cfg["source"]
    if s["type"] == "serial":
        return SerialSource(s["serial_port"], s["baud"], cfg["device_id"])
    sim = MotorCurrentSimulator(rated_rms=s.get("rated_rms", 0.8), seed=seed, fs=cfg["signal"]["fs"],
                                events=events)
    return SimulatedSource(sim, seconds, cfg["device_id"], start_ts=start_ts,
                           realtime=s.get("realtime", False) if realtime is None else realtime)


def make_pipeline(cfg, store=None, monitoring: bool = True, quiet: bool = False) -> Pipeline:
    sig = cfg["signal"]
    return Pipeline(
        cleaner=DespikeCleaner(sig["mode"]),
        summarizer=CurrentSummarizer(sig["mode"], sig["fs"], sig["mains_hz"]),
        store=store or make_store(cfg),
        load_type=cfg["load_type"],
        registry=make_registry(cfg) if monitoring else None,
        fusion=make_fusion(cfg), policy=make_policy(cfg),
        diagnoser=RuleBasedDiagnoser(), notifier=make_notifier(cfg, quiet=quiet))


def parse_events(spec: str):
    """'friction@60:120,smps@300:90' -> [FaultEvent(...), ...]"""
    out = []
    for part in filter(None, (spec or "").split(",")):
        kind, rest = part.split("@")
        start, dur = rest.split(":")
        out.append(FaultEvent(kind.strip(), float(start), float(dur)))
    return out
