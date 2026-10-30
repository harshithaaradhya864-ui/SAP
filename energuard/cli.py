"""Command-line entry point:  python -m energuard <command> [options]

  record        record healthy running (no models yet)
  train         train model v1 from the healthy recording and activate it
  run           live monitoring (simulated or real Pico)
  alerts        list alerts
  feedback      mark an alert as a real fault or a false alarm
  retrain       retrain with feedback; deploy only if better
  versions      list model versions
  evaluate      simulated test run with known faults -> report + chart
  simulate-feedback   (simulation) label alerts automatically from the known truth
  demo          everything above, end to end, on a fresh database
  telegram-feedback   save Telegram button presses as feedback
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import pandas as pd

from . import factory as F
from .evaluation import ablation, plot, save_report, score_alerts
from .interfaces import Feedback
from .simulator import evaluation_scenario
from .trainer import build_retraining_data, compare

SIM_EPOCH = 1_767_225_600.0   # 1 Jan 2026: fixed start time for simulated runs


def _next_ts(store) -> float:
    df = store.readings()
    return float(df["ts"].max()) + 1 if len(df) else SIM_EPOCH


def _healthy(store, device_id):
    df = store.readings(device_id=device_id)
    return df[df["model_version"].isna()].reset_index(drop=True)


# ------------------------------------------------------------------------------ commands
def cmd_record(cfg, a):
    store = F.make_store(cfg)
    secs = int(a.minutes * 60)
    print(f"Recording {a.minutes} min of healthy running...")
    src = F.make_source(cfg, secs, events=[], seed=a.seed, start_ts=_next_ts(store))
    stats = F.make_pipeline(cfg, store, monitoring=False).run(src, max_seconds=secs if cfg["source"]["type"] == "serial" else None,
                                                               progress_every=600)
    print(f"Saved {stats.seconds} seconds of healthy data.")


def cmd_train(cfg, a):
    store = F.make_store(cfg)
    healthy = _healthy(store, cfg["device_id"])
    print(f"Training on {len(healthy)} s of healthy data...")
    t0 = time.time()
    bundle = F.make_trainer(cfg).train(healthy, note=a.note or "initial training")
    reg = F.make_registry(cfg)
    v = reg.register(bundle, {"train_seconds": round(time.time() - t0, 1)})
    reg.activate(v)
    print(f"Model v{v} trained in {time.time() - t0:.0f}s with {[d.name for d in bundle.detectors]} and activated.")


def cmd_run(cfg, a):
    store = F.make_store(cfg)
    events = evaluation_scenario() if a.scenario else F.parse_events(a.faults)
    secs = int(a.minutes * 60)
    src = F.make_source(cfg, secs, events=events, seed=a.seed, start_ts=_next_ts(store),
                        realtime=True if a.realtime else None)
    pipe = F.make_pipeline(cfg, store, quiet=a.quiet)
    if not pipe.monitoring:
        raise SystemExit("No active model. Run `record` and `train` first.")
    print(f"Monitoring for {a.minutes} min" + (f" with faults: {a.faults}" if a.faults else "") + " ...")
    stats = pipe.run(src, progress_every=a.progress)
    print(f"Done: {stats.seconds} s monitored, {len(stats.alerts)} alerts.")
    return stats


def cmd_alerts(cfg, a):
    al = F.make_store(cfg).alerts(cfg["device_id"])
    fb = F.make_store(cfg).feedback()
    if al.empty:
        print("No alerts.")
        return
    al = al.merge(fb[["alert_id", "label"]], on="alert_id", how="left")
    al["time"] = pd.to_datetime(al["ts"], unit="s").dt.strftime("%m-%d %H:%M:%S")
    print(al[["alert_id", "time", "severity", "cause", "fused_score", "label"]].to_string(index=False))


def cmd_feedback(cfg, a):
    if a.label not in ("fault", "false_alarm"):
        raise SystemExit("label must be 'fault' or 'false_alarm'")
    F.make_store(cfg).write_feedback(Feedback(a.alert_id, cfg["device_id"], a.label, time.time(), a.note or ""))
    print(f"Saved: {a.alert_id} -> {a.label}")


def cmd_simulate_feedback(cfg, a):
    """Act as the maintenance user: compare each alert with the simulator's truth."""
    store = F.make_store(cfg)
    r = store.readings(cfg["device_id"])
    fb = store.feedback()
    done = set(fb["alert_id"]) if len(fb) else set()
    n = {"fault": 0, "false_alarm": 0}
    for _, al in store.alerts(cfg["device_id"]).iterrows():
        if al["alert_id"] in done:
            continue
        near = r[(r["ts"] >= al["ts"] - 30) & (r["ts"] <= al["ts"])]
        label = "fault" if near["truth"].notna().any() else "false_alarm"
        store.write_feedback(Feedback(al["alert_id"], cfg["device_id"], label, al["ts"] + 600, "simulated user"))
        n[label] += 1
    print(f"Simulated feedback: {n['fault']} real faults, {n['false_alarm']} false alarms.")


def cmd_retrain(cfg, a):
    store, reg = F.make_store(cfg), F.make_registry(cfg)
    fusion, policy = F.make_fusion(cfg), F.make_policy(cfg)
    data = build_retraining_data(store, _healthy(store, cfg["device_id"]), cfg["device_id"])
    print(f"Retraining on {len(data['healthy_train'])} s of trusted healthy data "
          f"({data['n_false']} false-alarm periods added as normal, {data['n_fault']} confirmed faults kept for testing)...")
    current = reg.load()
    candidate = F.make_trainer(cfg).train(data["healthy_train"], note=f"retrained from v{current.version} with feedback")
    m_old = compare(current, data, fusion, policy)
    m_new = compare(candidate, data, fusion, policy)
    better = (m_new["faults_caught"] >= m_old["faults_caught"] and m_new["false_alarms"] < m_old["false_alarms"]) or \
             (m_new["faults_caught"] > m_old["faults_caught"] and m_new["false_alarms"] <= m_old["false_alarms"])
    v = reg.register(candidate, {**m_new, "compared_with": current.version, "accepted": better})
    print(f"  current v{current.version}: {m_old}")
    print(f"  new     v{v}: {m_new}")
    if better:
        reg.activate(v)
        print(f"v{v} is better and is now active (the live pipeline picks it up within a minute).")
    else:
        print(f"v{v} is not better; v{current.version} stays active.")
    return {"old": m_old, "new": m_new, "new_version": v, "accepted": better}


def cmd_versions(cfg, a):
    for v in F.make_registry(cfg).versions():
        print(("* " if v["active"] else "  ") + f"v{v['version']}  {v.get('note', '')}  {v['metrics']}")


def cmd_evaluate(cfg, a):
    store = F.make_store(cfg)
    events = evaluation_scenario()
    secs = int(max(e.start + e.duration for e in events))
    start = _next_ts(store)
    print(f"Evaluation run: {secs // 60} min with 4 injected faults, 3 harmless start-ups and a harmless load change...")
    src = F.make_source(cfg, secs, events=events, seed=a.seed, start_ts=start, realtime=False)
    pipe = F.make_pipeline(cfg, store, quiet=True)
    stats = pipe.run(src, progress_every=a.progress)
    r = store.readings(cfg["device_id"], start=start)
    al = store.alerts(cfg["device_id"])
    al = al[al["ts"] >= start]
    live = score_alerts(r, list(al["ts"]))
    report = {"model_version": sorted(stats.model_versions), "live": live,
              "ablation": ablation(pipe.bundle, r, F.make_fusion(cfg), F.make_policy(cfg)),
              "alerts": al[["alert_id", "ts", "severity", "cause", "fused_score"]].to_dict("records")}
    os.makedirs(a.out, exist_ok=True)
    save_report(report, os.path.join(a.out, "evaluation_report.json"))
    names = {"startup": "start-up", "load_change": "new load (harmless)"}
    harmless = [(e.start, names[e.kind]) for e in events if e.kind in names]
    plot(r, al, os.path.join(a.out, "evaluation_chart.png"), cfg["alerts"]["threshold"], harmless)
    _print_report(report)
    print(f"\nReport and chart saved in {a.out}/")
    return report


def _print_report(rep):
    live = rep["live"]
    print("\nFault detection (live run):")
    for f in live["faults"]:
        d = f"caught after {f['delay_s']} s" if f["detected"] else "MISSED"
        print(f"  {f['fault']:<11} ({f['duration_s']:>4} s): {d}")
    print(f"  false alarms: {live['false_alarms']} in {live['healthy_hours']} h of healthy running "
          f"({live['false_alarms_per_hour']}/h)")
    print("\nWhy two models (same data replayed):")
    for k, v in rep["ablation"].items():
        delays = {f["fault"]: f["delay_s"] for f in v["faults"]}
        print(f"  {k:<24} caught {v['faults_detected']}/{v['faults_total']}, false alarms {v['false_alarms']}, delays {delays}")
    print("\nAlerts:")
    for al in rep["alerts"]:
        print(f"  {al['alert_id']}  {al['severity']:<6} {al['cause']}")


def cmd_demo(cfg, a):
    """Fresh, complete run of the whole system on simulated data."""
    for p in (cfg["store"]["sqlite_path"], cfg["models"]["root"]):
        if os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.exists(p):
            os.remove(p)
    ns = argparse.Namespace
    print("\n=== 1. Record healthy running ===")
    cmd_record(cfg, ns(minutes=a.healthy_minutes, seed=1))
    print("\n=== 2. Train model v1 (Isolation Forest + LSTM autoencoder) ===")
    cmd_train(cfg, ns(note="initial training"))
    print("\n=== 3. Live monitoring test with injected faults ===")
    rep = cmd_evaluate(cfg, ns(seed=2, out=a.out, progress=1200))
    print("\n=== 4. Maintenance user labels every alert ===")
    cmd_simulate_feedback(cfg, ns())
    print("\n=== 5. Retrain with feedback (MLOps) ===")
    rt = cmd_retrain(cfg, ns())
    print("\n=== 6. Model versions ===")
    cmd_versions(cfg, ns())
    save_report({"evaluation": rep, "retraining": rt}, os.path.join(a.out, "demo_summary.json"))


def cmd_telegram_feedback(cfg, a):
    import urllib.request
    t = cfg["notify"]["telegram"]
    store, offset = F.make_store(cfg), 0
    base = f"https://api.telegram.org/bot{t['bot_token']}"
    print("Listening for Telegram feedback buttons (Ctrl+C to stop)...")
    while True:
        try:
            res = json.load(urllib.request.urlopen(f"{base}/getUpdates?timeout=30&offset={offset}", timeout=40))
        except Exception as e:
            print(f"[telegram] {e}; retrying")
            time.sleep(5)
            continue
        for u in res.get("result", []):
            offset = u["update_id"] + 1
            cb = u.get("callback_query")
            if cb and cb.get("data", "").startswith("fb|"):
                _, alert_id, device, label = cb["data"].split("|")
                store.write_feedback(Feedback(alert_id, device, label, time.time(), "telegram"))
                urllib.request.urlopen(f"{base}/answerCallbackQuery?callback_query_id={cb['id']}&text=Saved", timeout=10)
                print(f"Saved {alert_id} -> {label}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="energuard", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("record"); s.add_argument("--minutes", type=float, default=45); s.add_argument("--seed", type=int, default=1)
    s = sub.add_parser("train"); s.add_argument("--note", default="")
    s = sub.add_parser("run")
    s.add_argument("--minutes", type=float, default=10)
    s.add_argument("--faults", default="", help="e.g. friction@120:90,smps@400:60 (kind@start_s:duration_s)")
    s.add_argument("--scenario", action="store_true", help="use the built-in evaluation fault timeline")
    s.add_argument("--realtime", action="store_true"); s.add_argument("--quiet", action="store_true")
    s.add_argument("--seed", type=int, default=7); s.add_argument("--progress", type=int, default=60)
    sub.add_parser("alerts")
    s = sub.add_parser("feedback"); s.add_argument("--alert-id", required=True)
    s.add_argument("--label", required=True, choices=["fault", "false_alarm"]); s.add_argument("--note", default="")
    sub.add_parser("simulate-feedback")
    sub.add_parser("retrain")
    sub.add_parser("versions")
    s = sub.add_parser("evaluate"); s.add_argument("--seed", type=int, default=2)
    s.add_argument("--out", default="results"); s.add_argument("--progress", type=int, default=600)
    s = sub.add_parser("demo"); s.add_argument("--healthy-minutes", type=float, default=45); s.add_argument("--out", default="results")
    sub.add_parser("telegram-feedback")

    a = p.parse_args(argv)
    cfg = F.load_config(a.config)
    {"record": cmd_record, "train": cmd_train, "run": cmd_run, "alerts": cmd_alerts, "feedback": cmd_feedback,
     "simulate-feedback": cmd_simulate_feedback, "retrain": cmd_retrain, "versions": cmd_versions,
     "evaluate": cmd_evaluate, "demo": cmd_demo, "telegram-feedback": cmd_telegram_feedback}[a.cmd](cfg, a)


if __name__ == "__main__":
    main()
