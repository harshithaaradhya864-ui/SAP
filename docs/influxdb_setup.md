# Moving from SQLite to InfluxDB + Grafana on the Raspberry Pi

The simulation uses SQLite so it runs anywhere with no setup. On the
Raspberry Pi 4 (64-bit Raspberry Pi OS), switch to InfluxDB 2.x and Grafana.
No Python code changes are needed: only `config.yaml`.

## 1. Install InfluxDB 2.x

```bash
curl -fsSL https://repos.influxdata.com/influxdata-archive.key | sudo gpg --dearmor -o /usr/share/keyrings/influxdata.gpg
echo "deb [signed-by=/usr/share/keyrings/influxdata.gpg] https://repos.influxdata.com/debian stable main" | sudo tee /etc/apt/sources.list.d/influxdata.list
sudo apt update && sudo apt install -y influxdb2 influxdb2-cli
sudo systemctl enable --now influxdb
```

## 2. Create the organisation and the three buckets

```bash
influx setup --org energuard --bucket raw_1s --retention 7d \
  --username admin --password <choose-one> --force
influx bucket create --name agg_1m --retention 365d --org energuard
influx bucket create --name events --retention 365d --org energuard
influx auth create --org energuard --all-access     # copy the token it prints
```

Retention rules (7 days, 365 days) make InfluxDB delete old rows itself.

## 3. Downsampling task (raw_1s -> agg_1m every minute)

Save as `agg_1m.flux` and run `influx task create --org energuard -f agg_1m.flux`:

```flux
option task = {name: "energuard_agg_1m", every: 1m}

data = from(bucket: "raw_1s")
  |> range(start: -2m, stop: now())
  |> filter(fn: (r) => r._measurement == "current")

data |> filter(fn: (r) => r._field == "rms") |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
     |> set(key: "_field", value: "rms_mean") |> set(key: "_measurement", value: "current_1m")
     |> to(bucket: "agg_1m")
data |> filter(fn: (r) => r._field == "peak") |> aggregateWindow(every: 1m, fn: max, createEmpty: false)
     |> set(key: "_field", value: "peak_max") |> set(key: "_measurement", value: "current_1m")
     |> to(bucket: "agg_1m")
data |> filter(fn: (r) => r._field == "fused_score") |> aggregateWindow(every: 1m, fn: max, createEmpty: false)
     |> set(key: "_field", value: "score_max") |> set(key: "_measurement", value: "current_1m")
     |> to(bucket: "agg_1m")
data |> filter(fn: (r) => r._field == "fused_score") |> map(fn: (r) => ({r with _value: if r._value > 0.5 then 1 else 0}))
     |> aggregateWindow(every: 1m, fn: sum, createEmpty: false)
     |> set(key: "_field", value: "anomaly_seconds") |> set(key: "_measurement", value: "current_1m")
     |> to(bucket: "agg_1m")
```

## 4. Point EnerGuard at InfluxDB

```yaml
store:
  type: influx
  influx:
    url: http://localhost:8086
    token: <token from step 2>
    org: energuard
```

```bash
pip install influxdb-client
```

## 5. Grafana dashboard

```bash
sudo apt install -y grafana        # from the Grafana apt repository
sudo systemctl enable --now grafana-server
```

Open `http://<pi-address>:3000`, add InfluxDB as a data source (Flux,
organisation `energuard`, the token above) and create panels:

| Panel | Bucket | Field |
|---|---|---|
| Live current | raw_1s | rms, peak |
| Anomaly score | raw_1s | if_score, ae_score, fused_score (threshold line at 0.5) |
| Long-term trend | agg_1m | rms_mean, score_max |
| Alerts | events / alerts | annotations on the graphs |
