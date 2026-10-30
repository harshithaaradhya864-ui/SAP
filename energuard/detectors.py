"""ML / DL anomaly detectors.

IsolationForestDetector (ML) - scores each second on its own (plus a healthy
                               range check); fast, catches sudden faults.
LSTMAutoencoderDetector (DL) - looks at how the current moved over the last
                               60 seconds and tries to rebuild that pattern;
                               catches new rhythms (e.g. the slow ripple of a
                               worn bearing) and sudden steps.

Both are trained on healthy data only and both return a 0-1 score:
0 = typical healthy, 0.5 = just above the 99th percentile of healthy data
(a small safety margin), 1 = far beyond.
"""
from __future__ import annotations

import json
import os
from collections import deque

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .interfaces import AnomalyDetector


MARGIN = 0.25   # safety gap above the 99th percentile, as a share of the healthy spread


class ScoreScaler:
    """Maps a detector's raw score to the shared 0-1 scale using healthy data."""

    def __init__(self, median: float = 0.0, p99: float = 1.0, margin: float = MARGIN,
                 far: float = None):
        self.median, self.p99, self.margin = median, p99, margin
        self.far = far          # raw score that maps to 1.0 (default: same slope as below the edge)

    def fit(self, raw: np.ndarray, far: float = None) -> "ScoreScaler":
        self.median = float(np.median(raw))
        self.p99 = float(np.percentile(raw, 99))
        if self.p99 - self.median < 1e-9:
            self.p99 = self.median + 1e-6
        self.far = far
        return self

    @property
    def edge(self) -> float:
        """Raw score that maps to 0.5: just above the 99th percentile of healthy data."""
        return self.p99 + self.margin * (self.p99 - self.median)

    def __call__(self, raw: float) -> float:
        e, m = self.edge, self.median
        if raw <= e:
            return float(max(0.0, 0.5 * (raw - m) / (e - m)))
        far = self.far if self.far is not None and self.far > e else 2 * e - m
        return float(min(1.0, 0.5 + 0.5 * (raw - e) / (far - e)))

    def to_dict(self):
        return {"median": self.median, "p99": self.p99, "margin": self.margin, "far": self.far}


# ============================================================================ ML
class IsolationForestDetector(AnomalyDetector):
    name = "if"

    def __init__(self, n_estimators: int = 200, seed: int = 42):
        self.n_estimators = n_estimators
        self.seed = seed
        self.scaler = StandardScaler()
        self.model = IsolationForest(n_estimators=n_estimators, random_state=seed)
        self.calib = ScoreScaler()
        self.lo = self.hi = None          # healthy range of each feature (range check)

    def _raw(self, X: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(self.scaler.transform(X))

    def fit(self, X, X_calib):
        self.model.fit(self.scaler.fit_transform(X))
        # Isolation Forest scores saturate: any point outside the healthy cloud
        # gets about the same raw score. Probe that ceiling with points far
        # outside the data so the 0.5-1 range is used properly.
        mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
        signs = np.array(np.meshgrid(*[[-1, 1]] * len(mu))).reshape(len(mu), -1).T
        probes = mu + 30 * sd * signs                 # far outside on every feature at once
        far = float(np.max(self._raw(probes)))
        self.calib.fit(self._raw(X_calib), far=far)
        H = np.vstack([X, X_calib])
        self.lo, self.hi = np.percentile(H, 0.5, axis=0), np.percentile(H, 99.5, axis=0)

    def _range_score(self, X: np.ndarray) -> np.ndarray:
        """Range check. Isolation Forest cannot tell 'just outside' the healthy
        data from 'far outside' (both land in the same tree leaves), so each
        feature is also compared with its healthy range (0.5-99.5 percentile).
        10 % of the range beyond the edge -> 0.5; a whole range beyond -> 1."""
        spread = np.maximum(self.hi - self.lo, 1e-9)
        e = np.max(np.maximum(0, np.maximum(X - self.hi, self.lo - X)) / spread, axis=1)
        return np.where(e < 0.1, 5 * e, np.minimum(1.0, 0.5 + 0.5 * (e - 0.1) / 0.9))

    def score(self, x):
        return float(self.score_batch(np.asarray(x, dtype=float).reshape(1, -1))[0])

    def score_batch(self, X):
        X = np.asarray(X, dtype=float)
        forest = np.array([self.calib(r) for r in self._raw(X)])
        return np.maximum(forest, self._range_score(X))

    def score_series(self, X):
        return self.score_batch(X)

    def reset(self):
        pass

    def save(self, folder):
        joblib.dump({"scaler": self.scaler, "model": self.model, "calib": self.calib.to_dict(),
                     "lo": self.lo, "hi": self.hi},
                    os.path.join(folder, "isolation_forest.joblib"))

    def load(self, folder):
        d = joblib.load(os.path.join(folder, "isolation_forest.joblib"))
        self.scaler, self.model = d["scaler"], d["model"]
        self.calib = ScoreScaler(**d["calib"])
        self.lo, self.hi = d["lo"], d["hi"]


# ============================================================================ DL
CLIP = 5.0   # autoencoder input limit, in healthy standard deviations


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _build_net(n_features: int, hidden: int, latent: int):
    import torch.nn as nn

    class LSTMAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
            self.to_latent = nn.Linear(hidden, latent)
            self.from_latent = nn.Linear(latent, hidden)
            self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
            self.output = nn.Linear(hidden, n_features)

        def forward(self, x):
            _, (h, _) = self.encoder(x)
            z = self.to_latent(h[-1])                                  # compressed summary
            d = self.from_latent(z).unsqueeze(1).repeat(1, x.size(1), 1)
            y, _ = self.decoder(d)
            return self.output(y)                                      # rebuilt sequence

    return LSTMAutoencoder()


class LSTMAutoencoderDetector(AnomalyDetector):
    name = "ae"

    def __init__(self, seq_len: int = 60, hidden: int = 32, latent: int = 8, epochs: int = 30,
                 batch_size: int = 64, lr: float = 1e-3, stride: int = 2, seed: int = 42, trim: float = 0.1):
        self.seq_len, self.hidden, self.latent = seq_len, hidden, latent
        self.trim = trim
        self.epochs, self.batch_size, self.lr, self.stride, self.seed = epochs, batch_size, lr, stride, seed
        self.mean = None
        self.std = None
        self.net = None
        self.calib = ScoreScaler()
        self.history = deque(maxlen=seq_len)

    # ------------------------------------------------------------ helpers
    def _norm(self, X):
        return (np.asarray(X, dtype=np.float32) - self.mean) / self.std

    def _sequences(self, Xn, stride):
        idx = range(0, len(Xn) - self.seq_len + 1, stride)
        return self._center(np.stack([Xn[i:i + self.seq_len] for i in idx]))

    @staticmethod
    def _center(seqs):
        """Remove each sequence's own typical (median) level. The autoencoder then judges
        the rhythm of the last minute (how the current moves), not the level,
        which the Isolation Forest already checks every second."""
        seqs = np.clip(seqs, -CLIP, CLIP)                       # one huge second cannot dominate
        return (seqs - np.median(seqs, axis=1, keepdims=True)).astype(np.float32)

    def _errors(self, seqs):
        """Rebuild error of each sequence: the per-second errors are averaged
        after dropping the worst 10 %, so a 1-2 second blip (a motor start-up)
        does not count as a changed pattern, while a real change that affects
        the whole minute does."""
        import torch
        self.net.eval()
        out = []
        drop = max(1, int(round(self.seq_len * self.trim)))
        with torch.no_grad():
            for i in range(0, len(seqs), 256):
                b = torch.from_numpy(seqs[i:i + 256])
                step_err = ((self.net(b) - b) ** 2).mean(dim=2)          # (batch, seq_len)
                kept = torch.sort(step_err, dim=1).values[:, :-drop]
                out.append(kept.mean(dim=1).numpy())
        return np.concatenate(out)

    # ------------------------------------------------------------ contract
    def fit(self, X, X_calib):
        import torch

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.mean = X.mean(axis=0).astype(np.float32)
        self.std = (X.std(axis=0) + 1e-6).astype(np.float32)
        seqs = self._sequences(self._norm(X), self.stride)
        self.net = _build_net(X.shape[1], self.hidden, self.latent)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        data = torch.from_numpy(seqs)
        for _ in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(data))
            for i in range(0, len(data), self.batch_size):
                b = data[perm[i:i + self.batch_size]]
                loss = ((self.net(b) - b) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.calib.fit(self._errors(self._sequences(self._norm(X_calib), 1)))
        self.reset()

    def score(self, x):
        self.history.append(self._norm(np.asarray(x, dtype=float).reshape(-1)))
        if len(self.history) < self.seq_len:
            return 0.0                                   # warming up: not enough history yet
        seq = self._center(np.stack(self.history)[None])
        return self.calib(float(self._errors(seq)[0]))

    def score_series(self, X):
        Xn = self._norm(X)
        out = np.zeros(len(Xn))
        if len(Xn) >= self.seq_len:
            errs = self._errors(self._sequences(Xn, 1))
            out[self.seq_len - 1:] = [self.calib(e) for e in errs]
        self.reset()
        return out

    def reset(self):
        self.history.clear()

    def save(self, folder):
        import torch
        torch.save(self.net.state_dict(), os.path.join(folder, "lstm_autoencoder.pt"))
        with open(os.path.join(folder, "lstm_autoencoder.json"), "w") as f:
            json.dump({"seq_len": self.seq_len, "hidden": self.hidden, "latent": self.latent, "trim": self.trim,
                       "mean": self.mean.tolist(), "std": self.std.tolist(),
                       "calib": self.calib.to_dict()}, f, indent=2)

    def load(self, folder):
        import torch
        with open(os.path.join(folder, "lstm_autoencoder.json")) as f:
            d = json.load(f)
        self.seq_len, self.hidden, self.latent = d["seq_len"], d["hidden"], d["latent"]
        self.trim = d.get("trim", 0.1)
        self.mean = np.array(d["mean"], dtype=np.float32)
        self.std = np.array(d["std"], dtype=np.float32)
        self.calib = ScoreScaler(**d["calib"])
        self.net = _build_net(len(self.mean), self.hidden, self.latent)
        self.net.load_state_dict(torch.load(os.path.join(folder, "lstm_autoencoder.pt"), weights_only=True))
        self.net.eval()
        self.history = deque(maxlen=self.seq_len)


DETECTOR_TYPES = {"if": IsolationForestDetector, "ae": LSTMAutoencoderDetector}
