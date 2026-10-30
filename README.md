# EnerGuard

AI-driven real-time energy anomaly detection for motor loads.
Isolation Forest (ML) + LSTM autoencoder (DL), an offline AI diagnoser,
InfluxDB/SQLite storage and an MLOps retraining loop, structured with
SOLID principles.

This version runs entirely in **software simulation**: a realistic motor
current simulator replaces the ACS712 → ADS1115 → Pico chain, so the whole
system can be built, tested and demonstrated before the hardware is wired.
Switching to the real sensor is one line in `config.yaml`.

---

## 1. Quick start (laptop, ~3 minutes)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install numpy pandas scikit-learn joblib pyyaml matplotlib pytest
pip install torch --index-url https://download.pytorch.org/whl/cpu   # small CPU build
python -m pytest -q                                      # 9 tests
python -m energuard demo                                 # the whole system, end to end
```

`demo` does everything on a fresh database:

1. records 45 minutes of healthy motor running
2. trains model v1 (Isolation Forest + LSTM autoencoder) and activates it
3. runs a 110-minute live test with 4 injected faults, 3 harmless motor
   start-ups and a harmless permanent load increase
4. plays the maintenance user: labels each alert *real fault* or *false alarm*
5. retrains with that feedback and deploys v2 only if it is better
6. lists the model versions

Results go to `results/` (`evaluation_chart.png`, `evaluation_report.json`,
`demo_summary.json`).

Dashboard (in a second terminal):

```bash
pip install streamlit
streamlit run dashboard/app.py
```

## 2. Results from the simulation

| Fault (injected) | Caught after |
|---|---|
| Shaft friction / overload (+25 % current) | 2 s |
| SMPS overload (peaky waveform) | 2 s |
| Loose wire (drop-outs) | 3 s |
| Bearing wear (slow drift + 12 s ripple over 30 min) | 195 s |
| False alarms in 75 min of healthy running | 1 (the harmless load increase) |
| Harmless motor start-ups (3) | 0 alerts |

Why two models? The same data replayed with each set-up:

| Set-up | Faults caught | False alarms | Bearing wear caught after |
|---|---|---|---|
| Isolation Forest only | 3 / 4 | 4 | 182 s |
| LSTM autoencoder only | 4 / 4 | 3 | 10 s |
| **Both (fused)** | **4 / 4** | **1** | 195 s |

MLOps: after the user marked the load increase as a false alarm, the
retrained v2 had 0 false alarms on the test data and still caught every
fault, so it was deployed automatically. Trade-off: once the heavier load
is accepted as normal, slow bearing wear is caught later (about 9 minutes
instead of 3), still well within the 30-minute fault.

Speed: about 13 ms of processing per second of data on a laptop, so a
Raspberry Pi 4 has plenty of headroom.

*These numbers come from simulated data. Re-measure them with the real
sensor before quoting them as final results.*

## 3. How it works

```
Pico / simulator ──► clean signal ──► Isolation Forest ─┐
   860 samples/s     offset, spikes   LSTM autoencoder ─┴► fusion ─► 3-second rule ─► AI diagnoser ─► alert
                                                                        │                              │
                                                     SQLite / InfluxDB ◄┘        maintenance user ◄────┘
                                                            ▲                          │ feedback
                                                     Trainer ─► ModelRegistry ◄────────┘
```

| Step | What happens | Code |
|---|---|---|
| Read | 860 current samples per second (simulated or from the Pico) | `sources.py`, `simulator.py`, `pico_firmware/main.py` |
| Clean | remove sensor offset and single-sample spikes | `processing.py` |
| Summarise | RMS, peak, steadiness (std of cycle RMS), crest factor | `processing.py` |
| Isolation Forest (ML) | is this second's level and shape normal? Plus a healthy-range check, because Isolation Forest alone cannot tell "just outside" from "far outside" | `detectors.py` |
| LSTM autoencoder (DL) | is the rhythm of the last 60 s normal? Each minute is centred on its own median, so the model judges how the current moves, not its level | `detectors.py` |
| Score fusion | `max(average, 0.8 × strongest score)`: both models together, or one very sure model alone (score > 0.625), can raise the alarm | `decision.py` |
| Alert rule | fused score > 0.5 for 3 s in a row; the same problem re-alerts only after 60 normal seconds | `decision.py` |
| AI diagnoser | offline rules → likely cause, severity, next step | `diagnoser.py` |
| Notify | console, Telegram (with Fault / False-alarm buttons), e-mail | `notifiers.py` |
| Store | `raw_1s` (7 days), `agg_1m` (1 year), `alerts`, `feedback` | `storage.py` |
| Train / retrain | train on healthy data; retrain with feedback; deploy only if better | `trainer.py` |
| Model registry | versioned models, one active, instant rollback | `registry.py` |

Every 0–1 score uses the same scale: 0 = typical healthy, 0.5 = just above
the 99th percentile of healthy data, 1 = far beyond.

### SOLID in the code

* **S** – each class has one job (`DespikeCleaner`, `IsolationForestDetector`, `ConsecutiveAlertPolicy`, …).
* **O** – new behaviour is a new class: e.g. add `SlackNotifier(Notifier)` without touching the pipeline.
* **L** – every `AnomalyDetector` keeps the same 0–1 contract, so detectors can be swapped or combined freely.
* **I** – small interfaces in `interfaces.py` (`SensorSource`, `Notifier`, `ReadingStore`, …).
* **D** – `Pipeline` depends only on interfaces; `factory.py` is the single place that picks the real classes from `config.yaml`.

## 4. Commands

```bash
python -m energuard record --minutes 45          # record healthy running
python -m energuard train                        # train v1 and activate it
python -m energuard run --minutes 10 --faults "friction@120:90,smps@400:60"
python -m energuard run --minutes 30 --realtime  # one second per second (for the dashboard)
python -m energuard alerts                       # list alerts with their labels
python -m energuard feedback --alert-id motor-1-1767228902 --label fault
python -m energuard retrain                      # retrain with feedback; deploy if better
python -m energuard versions
python -m energuard evaluate --seed 5            # test run with known faults -> results/
python -m energuard telegram-feedback            # save Telegram button presses
```

Fault kinds for `--faults`: `friction`, `smps`, `loose_wire`, `bearing`,
plus the harmless `startup` and `load_change` (format `kind@start_s:duration_s`).

## 5. Moving to the real hardware

1. Wire the circuit (see the comments at the top of `pico_firmware/main.py`).
   **Test on a low-voltage DC motor first; never work on mains wiring live.**
2. Flash MicroPython on the Pico and save `pico_firmware/main.py` to it as `main.py`.
   Keep the load switched off for the first 3 seconds (zero-offset calibration).
3. On the Raspberry Pi 4 (64-bit Raspberry Pi OS): `pip install -r requirements.txt`.
4. In `config.yaml` set `source.type: serial` (and `signal.mode: dc` for a DC motor).
5. `python -m energuard record --minutes 45`, then `train`, then `run --minutes 600`.
6. Optional: switch to InfluxDB + Grafana with `docs/influxdb_setup.md`.

Alerts to your phone: create a bot with Telegram's @BotFather, put the
token and your chat id in `config.yaml` (`notify.telegram`), set
`enabled: true`, and run `python -m energuard telegram-feedback` alongside
monitoring so the Fault / False-alarm buttons are saved.

## 6. Project layout

```
energuard/
  interfaces.py   abstract interfaces + data classes (SOLID core)
  simulator.py    motor/SMPS current simulator with fault injection
  sources.py      SimulatedSource, SerialSource (Pico)
  processing.py   DespikeCleaner, CurrentSummarizer
  detectors.py    IsolationForestDetector (ML), LSTMAutoencoderDetector (DL)
  decision.py     WeightedFusion, ConsecutiveAlertPolicy
  diagnoser.py    RuleBasedDiagnoser (AI agent, offline)
  notifiers.py    Console, Telegram, E-mail
  storage.py      SQLiteStore, InfluxStore
  registry.py     FileModelRegistry
  trainer.py      Trainer, retraining data, model comparison
  pipeline.py     the live loop
  factory.py      builds everything from config.yaml
  evaluation.py   metrics + charts for simulated runs
  cli.py          commands
pico_firmware/main.py   MicroPython for the Pico
dashboard/app.py        Streamlit dashboard + feedback form
docs/influxdb_setup.md  InfluxDB + Grafana on the Pi
tests/                  unit tests
config.yaml             all settings
```
