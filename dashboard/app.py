import os
import sys
import time

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from energuard import factory as F            # noqa: E402
from energuard.interfaces import Feedback      # noqa: E402

cfg = F.load_config(os.environ.get("ENERGUARD_CONFIG", "config.yaml"))
store = F.make_store(cfg)
dev = cfg["device_id"]

st.set_page_config(page_title="EnerGuard", layout="wide")
st.title("EnerGuard: motor current monitor")

minutes = st.sidebar.slider("Show last N minutes", 5, 240, 30)
auto = st.sidebar.checkbox("Auto-refresh every 5 s", value=True)

raw = store.readings(dev)
if raw.empty:
    st.info("No data yet. Run `python -m energuard demo` or start monitoring.")
    st.stop()

end = raw["ts"].max()
df = raw[raw["ts"] >= end - minutes * 60].copy()
df["time"] = pd.to_datetime(df["ts"], unit="s")
last = df.iloc[-1]

c1, c2, c3, c4 = st.columns(4)
c1.metric("RMS current", f"{last['rms']:.3f} A")
c2.metric("Peak", f"{last['peak']:.2f} A")
c3.metric("Crest factor", f"{last['crest_factor']:.2f}")
score = last["fused_score"]
c4.metric("Anomaly score", "recording" if pd.isna(score) else f"{score:.2f}",
          delta=None if pd.isna(score) else ("ABNORMAL" if score > cfg["alerts"]["threshold"] else "normal"),
          delta_color="inverse")

st.subheader("Current")
st.line_chart(df.set_index("time")[["rms", "peak"]], height=220)
st.subheader("Anomaly scores (alert line = %.2f)" % cfg["alerts"]["threshold"])
st.line_chart(df.set_index("time")[["if_score", "ae_score", "fused_score"]], height=220)

st.subheader("Alerts")
alerts = store.alerts(dev)
fb = store.feedback()
if alerts.empty:
    st.write("No alerts.")
else:
    alerts = alerts.merge(fb[["alert_id", "label"]], on="alert_id", how="left").sort_values("ts", ascending=False)
    alerts["time"] = pd.to_datetime(alerts["ts"], unit="s")
    st.dataframe(alerts[["time", "severity", "cause", "fused_score", "label", "alert_id"]],
                 width="stretch", hide_index=True)

    st.subheader("Feedback form")
    open_alerts = alerts[alerts["label"].isna()]
    if open_alerts.empty:
        st.write("Every alert has been labelled.")
    else:
        pick = st.selectbox("Alert", open_alerts["alert_id"])
        row = open_alerts[open_alerts["alert_id"] == pick].iloc[0]
        st.write(f"**{row['cause']}** ({row['severity']})")
        st.caption(row["diagnosis"])
        st.write(f"Suggested action: {row['action']}")
        note = st.text_input("Note (optional)")
        b1, b2 = st.columns(2)
        if b1.button("Real fault"):
            store.write_feedback(Feedback(pick, dev, "fault", time.time(), note))
            st.rerun()
        if b2.button("False alarm"):
            store.write_feedback(Feedback(pick, dev, "false_alarm", time.time(), note))
            st.rerun()

st.subheader("Model versions")
vers = pd.DataFrame(F.make_registry(cfg).versions())
if not vers.empty:
    vers["trained_at"] = pd.to_datetime(vers["trained_at"], unit="s").dt.strftime("%Y-%m-%d %H:%M")
    m = pd.json_normalize(vers.pop("metrics").tolist())
    keep = [c for c in ("faults_caught", "faults_total", "false_alarms", "accepted") if c in m]
    vers = pd.concat([vers, m[keep]], axis=1)
st.dataframe(vers, width="stretch", hide_index=True)

if auto:
    time.sleep(5)
    st.rerun()
