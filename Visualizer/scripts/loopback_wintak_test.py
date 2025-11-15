"""Windows loopback harness to prove PyTak export without WinTAK."""
from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run swarm-visualizer and capture CoT packets locally.")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--max-ticks", type=int, default=5)
    parser.add_argument("--step-delay", type=float, default=0.5)
    parser.add_argument("--origin-lat", type=float, default=38.8895)
    parser.add_argument("--origin-lon", type=float, default=-77.0353)
    parser.add_argument("--altitude-ft", type=int, default=200)
    parser.add_argument("--cot-host", default="127.0.0.1")
    parser.add_argument("--cot-port", type=int, default=6969)
    parser.add_argument("--expected", type=int, default=5, help="Packets to capture before stopping")
    parser.add_argument("--timeout", type=float, default=15.0, help="Seconds to wait for packets")
    return parser.parse_args()


def start_listener(host: str, port: int, expected: int, timeout: float) -> Tuple[threading.Thread, List[bytes], threading.Event]:
    messages: List[bytes] = []
    stop_event = threading.Event()

    def _loop() -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind((host, port))
            sock.settimeout(0.5)
            deadline = time.time() + timeout
            while not stop_event.is_set() and time.time() < deadline:
                try:
                    data, addr = sock.recvfrom(65535)
                except socket.timeout:
                    continue
                messages.append(data)
                print(f"[listener] Received {len(data)} bytes from {addr}")
                if len(messages) >= expected:
                    print("[listener] Expected packet count reached; waiting for streamer to finish...")

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    return thread, messages, stop_event


def run_cli(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "-m",
        "swarm_visualizer.cli",
        "--iterations",
        str(args.iterations),
        "--max-ticks",
        str(args.max_ticks),
        "--step-delay",
        str(args.step_delay),
        "--cot-endpoint",
        f"udp://{args.cot_host}:{args.cot_port}",
        "--origin-lat",
        str(args.origin_lat),
        "--origin-lon",
        str(args.origin_lon),
        "--altitude-ft",
        str(args.altitude_ft),
    ]
    print("[cli]", " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    listener_thread, messages, stop_event = start_listener(
        args.cot_host, args.cot_port, args.expected, args.timeout
    )
    try:
        run_cli(args)
    finally:
        stop_event.set()
        listener_thread.join()
    print(f"Captured {len(messages)} CoT packets locally.")


if __name__ == "__main__":
    main()
