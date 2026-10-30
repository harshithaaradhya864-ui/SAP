# EnerGuard - Raspberry Pi Pico firmware (MicroPython)
#
# Reads the ACS712 current sensor through the ADS1115 ADC at 860 samples/s,
# converts each reading to amps and sends one line per second to the
# Raspberry Pi over USB:   W,<pico_ms>,<n>,<a1>,<a2>,...,<an>
#
# Wiring (see the Circuit Connections slide):
#   ADS1115 VDD -> Pico 3V3 OUT     ADS1115 GND  -> GND
#   ADS1115 SDA -> GP4 (pin 6)      ADS1115 SCL  -> GP5 (pin 7)
#   ADS1115 ADDR -> GND (0x48)      ADS1115 A0   -> ACS712 OUT via 10k/20k divider
#   ACS712 VCC -> VBUS (5 V)        ACS712 IP+/IP- in series with the load
#
# Copy this file to the Pico as main.py (Thonny: File > Save as > Raspberry Pi Pico).
# On first start, keep the load SWITCHED OFF for 3 seconds: the zero offset is measured.

import sys
import time
from machine import I2C, Pin

ADDR = 0x48
REG_CONV, REG_CFG = 0x00, 0x01
SENS_V_PER_A = 0.185          # ACS712-5A: 185 mV per amp
DIVIDER = 2 / 3               # 10k / 20k divider: ADC sees 2/3 of the sensor voltage
LSB_V = 4.096 / 32768         # ADS1115 at +-4.096 V range
FS = 860
CHUNK = FS                    # samples per line (one second)

i2c = I2C(0, sda=Pin(4), scl=Pin(5), freq=400_000)
led = Pin(25, Pin.OUT)

# Config: AIN0 vs GND, +-4.096 V, continuous mode, 860 SPS, comparator off
CFG = (0 << 15) | (0b100 << 12) | (0b001 << 9) | (0 << 8) | (0b111 << 5) | 0b00011
i2c.writeto_mem(ADDR, REG_CFG, bytes([CFG >> 8, CFG & 0xFF]))
time.sleep_ms(10)


def read_volts():
    b = i2c.readfrom_mem(ADDR, REG_CONV, 2)
    raw = (b[0] << 8) | b[1]
    if raw > 32767:
        raw -= 65536
    return raw * LSB_V / DIVIDER          # undo the divider: voltage at the ACS712 output


def calibrate_zero(seconds=3):
    """Average sensor voltage with NO load = zero-current offset (about 2.5 V)."""
    n, total = 0, 0.0
    end = time.ticks_add(time.ticks_ms(), seconds * 1000)
    while time.ticks_diff(end, time.ticks_ms()) > 0:
        total += read_volts()
        n += 1
        time.sleep_us(1100)
    return total / n


zero_v = calibrate_zero()
period_us = 1_000_000 // FS
buf = [0.0] * CHUNK

while True:
    led.toggle()
    t_next = time.ticks_us()
    for i in range(CHUNK):
        # I = (V - zero offset) / sensitivity
        buf[i] = (read_volts() - zero_v) / SENS_V_PER_A
        t_next = time.ticks_add(t_next, period_us)
        while time.ticks_diff(t_next, time.ticks_us()) > 0:
            pass
    sys.stdout.write("W,%d,%d," % (time.ticks_ms(), CHUNK))
    sys.stdout.write(",".join("%.4f" % a for a in buf))
    sys.stdout.write("\n")
