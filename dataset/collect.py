import serial
import csv
import time
from datetime import datetime

# Change this to your Pico W COM port
SERIAL_PORT = "COM12"

BAUD_RATE = 115200

import os

script_directory = os.path.dirname(os.path.abspath(__file__))

filename = os.path.join(
    script_directory,
    "motor_current_dataset.csv"
)

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)

time.sleep(2)

print("Starting data collection...")
print("Press CTRL+C to stop.")

with open(filename, "a", newline="") as file:

    writer = csv.writer(file)

    if file.tell() == 0:
        writer.writerow([
            "Timestamp",
            "Raw_ADC",
            "ADC_Voltage",
            "Sensor_Voltage",
            "Current_A"
        ])

    try:

        while True:

            line = ser.readline().decode(
                "utf-8", errors="ignore"
            ).strip()

            if not line:
                continue

            values = line.split(",")

            if len(values) != 5:
                continue

            timestamp = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )

            writer.writerow([
                timestamp,
                *values
            ])

            file.flush()

            print("Saved:", values)

    except KeyboardInterrupt:

        print("\nData collection stopped.")

    finally:

        ser.close()