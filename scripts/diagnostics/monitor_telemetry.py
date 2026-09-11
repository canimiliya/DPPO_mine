"""Sample GPU and host CPU telemetry while a child process is running."""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path


def cpu_times() -> tuple[int, int]:
    fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
    values = [int(value) for value in fields]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values), idle


def gpu_sample() -> list[str]:
    query = ",".join(
        [
            "utilization.gpu",
            "memory.used",
            "memory.total",
            "clocks.sm",
            "power.draw",
            "temperature.gpu",
        ]
    )
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [part.strip() for part in result.stdout.strip().split(",")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "timestamp_unix",
        "child_pid",
        "cpu_total_percent",
        "gpu_util_percent",
        "gpu_memory_used_mib",
        "gpu_memory_total_mib",
        "gpu_sm_clock_mhz",
        "gpu_power_w",
        "gpu_temperature_c",
    ]
    previous_total, previous_idle = cpu_times()
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        while Path(f"/proc/{args.pid}").exists():
            time.sleep(args.interval)
            if not Path(f"/proc/{args.pid}").exists():
                break
            total, idle = cpu_times()
            total_delta = total - previous_total
            idle_delta = idle - previous_idle
            previous_total, previous_idle = total, idle
            cpu_percent = 100.0 * (1.0 - idle_delta / total_delta) if total_delta else 0.0
            try:
                gpu = gpu_sample()
            except (subprocess.CalledProcessError, FileNotFoundError):
                gpu = ["NA"] * 6
            writer.writerow([time.time(), args.pid, f"{cpu_percent:.3f}", *gpu])
            handle.flush()


if __name__ == "__main__":
    main()
