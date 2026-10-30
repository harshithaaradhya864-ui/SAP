"""Fast unit tests: python -m pytest -q"""
import numpy as np
import pytest

from energuard.decision import ConsecutiveAlertPolicy, WeightedFusion
from energuard.detectors import IsolationForestDetector, ScoreScaler
from energuard.diagnoser import RuleBasedDiagnoser
from energuard.interfaces import Alert, Feedback, Reading
from energuard.processing import CurrentSummarizer, DespikeCleaner
from energuard.registry import FileModelRegistry
from energuard.simulator import FaultEvent, MotorCurrentSimulator
from energuard.storage import SQLiteStore
from energuard.trainer import Trainer

CLEAN, SUM = DespikeCleaner("ac"), CurrentSummarizer("ac")


def summarize(sim, sec):
    return SUM.summarize(CLEAN.clean(sim.second(sec)))


def test_simulator_rms_matches_rating():
    f = summarize(MotorCurrentSimulator(rated_rms=0.8, seed=1), 0)
    assert 0.75 < f["rms"] < 0.85
    assert 1.25 < f["crest_factor"] < 1.7


def test_faults_change_the_right_feature():
    healthy = summarize(MotorCurrentSimulator(seed=3), 50)
    fr = summarize(MotorCurrentSimulator(seed=3, events=[FaultEvent("friction", 0, 100)]), 50)
    sm = summarize(MotorCurrentSimulator(seed=3, events=[FaultEvent("smps", 0, 100)]), 50)
    assert fr["rms"] > healthy["rms"] * 1.15
    assert sm["crest_factor"] > healthy["crest_factor"] * 1.15


def test_cleaner_removes_offset_and_spikes():
    x = np.sin(np.linspace(0, 100 * np.pi, 860)) + 0.3
    x[100] = 25.0
    y = CLEAN.clean(x)
    assert abs(y.mean()) < 0.01 and y.max() < 1.5


def test_score_scaler_maps_healthy_to_half():
    s = ScoreScaler(margin=0).fit(np.random.default_rng(0).normal(size=10000))
    assert s(s.median) == 0.0
    assert abs(s(s.p99) - 0.5) < 1e-9
    assert s(1e9) == 1.0


def test_fusion_renormalises_missing_detector():
    f = WeightedFusion({"if": 0.5, "ae": 0.5}, strong=0.8)
    assert f.fuse({"if": 0.6, "ae": 0.4}) == pytest.approx(0.5)      # average wins
    assert f.fuse({"if": 1.0, "ae": 0.0}) == pytest.approx(0.8)      # one very sure model can alert
    assert f.fuse({"if": 0.6, "ae": 0.0}) < 0.5                       # one unsure model cannot
    assert f.fuse({"if": 0.8}) == pytest.approx(0.8)                  # missing detector


def test_alert_policy_needs_three_seconds_and_clears():
    p = ConsecutiveAlertPolicy(0.5, 3, clear_after=5)
    assert [p.update(s, 0) for s in (0.9, 0.9, 0.1, 0.9, 0.9)] == [None] * 5   # spikes ignored
    assert p.update(0.9, 0) == 3                                               # third in a row
    assert p.update(0.9, 0) is None                                            # same problem: no repeat
    for _ in range(5):
        p.update(0.1, 0)
    assert [p.update(0.9, 0) for _ in range(3)][-1] == 3                       # new problem: new alert


def test_isolation_forest_flags_friction():
    sim = MotorCurrentSimulator(seed=1)
    X = np.array([list(summarize(sim, i).values()) for i in range(900)])
    d = IsolationForestDetector()
    d.fit(X[:720], X[720:])
    bad = list(summarize(MotorCurrentSimulator(seed=2, events=[FaultEvent("friction", 0, 60)]), 30).values())
    assert d.score(np.array(bad)) > 0.6


def test_store_and_registry_roundtrip(tmp_path):
    store = SQLiteStore(str(tmp_path / "t.db"))
    for i in range(130):
        store.write_reading(Reading("m", "motor", 60.0 * 100 + i, 0.8, 1.2, 0.006, 1.5, 0.1, 0.1, 0.1, 1))
    store.write_alert(Alert("a1", "m", 6050, 0.9, 3, "high", "x", "y", "z", 1))
    store.write_feedback(Feedback("a1", "m", "false_alarm", 6100))
    assert len(store.readings("m")) == 130
    assert store.downsample() >= 2
    assert store.feedback().iloc[0]["label"] == "false_alarm"

    sim = MotorCurrentSimulator(seed=1)
    import pandas as pd
    df = pd.DataFrame([summarize(sim, i) for i in range(700)])
    bundle = Trainer({"if": IsolationForestDetector}).train(df)
    reg = FileModelRegistry(str(tmp_path / "models"))
    v = reg.register(bundle, {})
    reg.activate(v)
    loaded = reg.load()
    x = df.iloc[0].to_numpy()
    assert loaded.version == 1
    assert loaded.detectors[0].score(x) == pytest.approx(bundle.detectors[0].score(x))


def test_diagnoser_recognises_signatures():
    base = {"rms": 0.8, "peak": 1.1, "std_dev": 0.006, "crest_factor": 1.35}
    normal = dict(rms=0.8, peak=1.1, std_dev=0.006, crest_factor=1.35, if_score=0.1, ae_score=0.1, fused_score=0.1)

    def mk(ae=0.5, **fault):
        rs = [Reading("m", "motor", i, **normal) for i in range(30)]
        rs += [Reading("m", "motor", 30 + i, **{**normal, "if_score": 0.9, "ae_score": ae, "fused_score": 0.8, **fault})
               for i in range(3)]
        return rs

    d = RuleBasedDiagnoser()
    assert d.diagnose(mk(rms=1.0, peak=1.37), base).cause.startswith("Mechanical")
    assert d.diagnose(mk(crest_factor=1.7), base).cause.startswith("SMPS")
    assert d.diagnose(mk(std_dev=0.05, rms=0.78), base).cause.startswith("Loose")
    slow = mk(ae=0.9, rms=0.82)
    for r in slow[-3:]:
        r.if_score = 0.4
    assert d.diagnose(slow, base).cause.startswith("Gradual")
