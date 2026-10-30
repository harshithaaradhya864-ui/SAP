"""Sensor sources: where one-second windows of current come from.

SimulatedSource - software simulation (no hardware needed)
SerialSource    - the real Raspberry Pi Pico over USB serial
Both keep the SensorSource contract, so the pipeline works with either.
"""
from __future__ import annotations

import time
from typing import Iterator, Optional

import numpy as np

from .interfaces import SensorSource, Window
from .simulator import MotorCurrentSimulator


class SimulatedSource(SensorSource):
    def __init__(self, simulator: MotorCurrentSimulator, seconds: int, device_id: str = "motor-1",
                 start_ts: Optional[float] = None, realtime: bool = False):
        self.sim = simulator
        self.seconds = seconds
        self.device_id = device_id
        self.start_ts = time.time() if start_ts is None else start_ts
        self.realtime = realtime

    def windows(self) -> Iterator[Window]:
        for sec in range(self.seconds):
            t0 = time.time()
            samples = self.sim.second(sec)
            yield Window(self.device_id, self.start_ts + sec, samples, truth=self.sim.active_fault(sec + 0.5))
            if self.realtime:
                time.sleep(max(0.0, 1.0 - (time.time() - t0)))


class SerialSource(SensorSource):
    """Reads windows sent by pico_firmware/main.py.

    Line format (one line per second):  W,<pico_ms>,<n>,<a1>,<a2>,...,<an>
    """

    def __init__(self, port: str = "/dev/ttyACM0", baud: int = 115200, device_id: str = "motor-1",
                 min_samples: int = 700):
        self.port = port
        self.baud = baud
        self.device_id = device_id
        self.min_samples = min_samples

    def windows(self) -> Iterator[Window]:
        import serial  # pyserial, only needed with real hardware

        with serial.Serial(self.port, self.baud, timeout=3) as ser:
            ser.reset_input_buffer()
            while True:
                line = ser.readline().decode("ascii", errors="ignore").strip()
                if not line.startswith("W,"):
                    continue
                parts = line.split(",")
                try:
                    n = int(parts[2])
                    samples = np.array(parts[3:3 + n], dtype=float)
                except (ValueError, IndexError):
                    continue                      # corrupted line: skip it
                if samples.size >= self.min_samples:
                    yield Window(self.device_id, time.time(), samples)
