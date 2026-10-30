"""ReadingStore implementations.

SQLiteStore - zero-setup file database, used for the simulation and fine on a Pi.
InfluxStore - InfluxDB 2.x, the production choice on the Raspberry Pi.
Both keep the same buckets/tables: raw_1s, agg_1m, alerts, feedback.
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import asdict
from typing import List, Optional

import pandas as pd

from .interfaces import Alert, Feedback, Reading, ReadingStore

DAY = 86400.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_1s (
    device_id TEXT, load_type TEXT, ts REAL,
    rms REAL, peak REAL, std_dev REAL, crest_factor REAL,
    if_score REAL, ae_score REAL, fused_score REAL, model_version INTEGER, truth TEXT);
CREATE INDEX IF NOT EXISTS raw_dev_ts ON raw_1s(device_id, ts);
CREATE TABLE IF NOT EXISTS agg_1m (
    device_id TEXT, ts REAL, rms_mean REAL, peak_max REAL, score_max REAL, anomaly_seconds INTEGER,
    PRIMARY KEY (device_id, ts));
CREATE TABLE IF NOT EXISTS alerts (
    alert_id TEXT PRIMARY KEY, device_id TEXT, ts REAL, fused_score REAL, duration_s INTEGER,
    severity TEXT, cause TEXT, action TEXT, diagnosis TEXT, model_version INTEGER);
CREATE TABLE IF NOT EXISTS feedback (
    alert_id TEXT, device_id TEXT, label TEXT, ts REAL, note TEXT);
"""


class SQLiteStore(ReadingStore):
    def __init__(self, path: str = "data/energuard.db", raw_days: float = 7, history_days: float = 365,
                 batch: int = 60):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.raw_days, self.history_days, self.batch = raw_days, history_days, batch
        self.con = sqlite3.connect(path, check_same_thread=False)
        self.con.executescript(SCHEMA)
        self._buf: List[tuple] = []

    # ---------------------------------------------------------------- writes
    def write_reading(self, r: Reading) -> None:
        self._buf.append((r.device_id, r.load_type, r.ts, r.rms, r.peak, r.std_dev, r.crest_factor,
                          r.if_score, r.ae_score, r.fused_score, r.model_version, r.truth))
        if len(self._buf) >= self.batch:
            self.flush()

    def flush(self) -> None:
        if self._buf:
            self.con.executemany("INSERT INTO raw_1s VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", self._buf)
            self.con.commit()
            self._buf.clear()

    def write_alert(self, a: Alert) -> None:
        self.flush()
        d = a.to_dict()
        self.con.execute("INSERT OR REPLACE INTO alerts VALUES (:alert_id,:device_id,:ts,:fused_score,:duration_s,"
                         ":severity,:cause,:action,:diagnosis,:model_version)", d)
        self.con.commit()

    def write_feedback(self, f: Feedback) -> None:
        self.con.execute("DELETE FROM feedback WHERE alert_id = ?", (f.alert_id,))
        self.con.execute("INSERT INTO feedback VALUES (:alert_id,:device_id,:label,:ts,:note)", asdict(f))
        self.con.commit()

    # ---------------------------------------------------------------- reads
    def readings(self, device_id=None, start=None, end=None) -> pd.DataFrame:
        self.flush()
        q, p = "SELECT * FROM raw_1s WHERE 1=1", []
        if device_id:
            q += " AND device_id = ?"; p.append(device_id)
        if start is not None:
            q += " AND ts >= ?"; p.append(start)
        if end is not None:
            q += " AND ts < ?"; p.append(end)
        return pd.read_sql_query(q + " ORDER BY ts", self.con, params=p)

    def alerts(self, device_id=None) -> pd.DataFrame:
        q, p = "SELECT * FROM alerts", []
        if device_id:
            q += " WHERE device_id = ?"; p.append(device_id)
        return pd.read_sql_query(q + " ORDER BY ts", self.con, params=p)

    def feedback(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM feedback ORDER BY ts", self.con)

    def agg_1m(self, device_id=None) -> pd.DataFrame:
        q, p = "SELECT * FROM agg_1m", []
        if device_id:
            q += " WHERE device_id = ?"; p.append(device_id)
        return pd.read_sql_query(q + " ORDER BY ts", self.con, params=p)

    # ---------------------------------------------------------------- maintenance
    def downsample(self) -> int:
        """Summarise every complete minute of raw_1s into agg_1m (what the
        InfluxDB task does every minute in production)."""
        self.flush()
        cur = self.con.execute("""
            INSERT OR REPLACE INTO agg_1m
            SELECT device_id, CAST(ts / 60 AS INTEGER) * 60.0 AS m,
                   AVG(rms), MAX(peak), MAX(COALESCE(fused_score, 0)),
                   SUM(CASE WHEN fused_score > 0.5 THEN 1 ELSE 0 END)
            FROM raw_1s
            WHERE ts >= COALESCE((SELECT MAX(ts) FROM agg_1m), 0)
            GROUP BY device_id, m
            HAVING COUNT(*) >= 50""")
        self.con.commit()
        return cur.rowcount

    def apply_retention(self, now: float) -> None:
        self.flush()
        self.con.execute("DELETE FROM raw_1s WHERE ts < ?", (now - self.raw_days * DAY,))
        for t in ("agg_1m", "alerts", "feedback"):
            self.con.execute(f"DELETE FROM {t} WHERE ts < ?", (now - self.history_days * DAY,))
        self.con.commit()

    def clear(self) -> None:
        self._buf.clear()
        for t in ("raw_1s", "agg_1m", "alerts", "feedback"):
            self.con.execute(f"DELETE FROM {t}")
        self.con.commit()


class InfluxStore(ReadingStore):
    """InfluxDB 2.x store for the Raspberry Pi (pip install influxdb-client).

    Buckets: raw_1s (7 d), agg_1m (365 d), events (365 d; measurements
    `alerts` and `feedback`). Downsampling runs inside InfluxDB as a Flux
    task (see docs/influxdb_setup.md), so downsample() does nothing here.
    """

    def __init__(self, url: str, token: str, org: str, raw_bucket: str = "raw_1s",
                 agg_bucket: str = "agg_1m", events_bucket: str = "events"):
        from influxdb_client import InfluxDBClient
        from influxdb_client.client.write_api import SYNCHRONOUS

        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()
        self.org = org
        self.raw, self.agg, self.events = raw_bucket, agg_bucket, events_bucket

    def _point(self, measurement, tags, fields, ts):
        from influxdb_client import Point, WritePrecision
        p = Point(measurement)
        for k, v in tags.items():
            p = p.tag(k, v)
        for k, v in fields.items():
            if v is not None:
                p = p.field(k, v)
        return p.time(int(ts * 1000), WritePrecision.MS)

    def write_reading(self, r: Reading) -> None:
        fields = {k: getattr(r, k) for k in ("rms", "peak", "std_dev", "crest_factor", "if_score",
                                              "ae_score", "fused_score", "model_version")}
        self.write_api.write(self.raw, self.org, self._point(
            "current", {"device_id": r.device_id, "load_type": r.load_type}, fields, r.ts))

    def write_alert(self, a: Alert) -> None:
        self.write_api.write(self.events, self.org, self._point(
            "alerts", {"device_id": a.device_id, "severity": a.severity, "cause": a.cause},
            {"alert_id": a.alert_id, "fused_score": a.fused_score, "duration_s": a.duration_s,
             "diagnosis": a.diagnosis, "action": a.action, "model_version": a.model_version}, a.ts))

    def write_feedback(self, f: Feedback) -> None:
        self.write_api.write(self.events, self.org, self._point(
            "feedback", {"device_id": f.device_id, "label": f.label},
            {"alert_id": f.alert_id, "note": f.note}, f.ts))

    def _query(self, bucket, measurement, start, end, device_id=None) -> pd.DataFrame:
        rng = f"start: {int(start) if start else 0}" + (f", stop: {int(end)}" if end else "")
        dev = f' and r.device_id == "{device_id}"' if device_id else ""
        flux = (f'from(bucket: "{bucket}") |> range({rng}) '
                f'|> filter(fn: (r) => r._measurement == "{measurement}"{dev}) '
                '|> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")')
        df = self.query_api.query_data_frame(flux, org=self.org)
        if isinstance(df, list):
            df = pd.concat(df) if df else pd.DataFrame()
        if not df.empty:
            df["ts"] = pd.to_datetime(df["_time"]).astype("int64") / 1e9
        return df

    def readings(self, device_id=None, start=None, end=None):
        return self._query(self.raw, "current", start, end, device_id)

    def alerts(self, device_id=None):
        return self._query(self.events, "alerts", None, None, device_id)

    def feedback(self):
        return self._query(self.events, "feedback", None, None)

    def downsample(self) -> int:
        return 0  # handled by the InfluxDB task

    def apply_retention(self, now: float) -> None:
        pass      # handled by bucket retention rules
