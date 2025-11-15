"""Loopback harness to validate PyTak TCP streaming on Windows."""
from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture CoT packets over TCP whilst running swarm-visualizer.")
    parser.add_argument("--cot-host", default="127.0.0.1")
    parser.add_argument("--cot-port", type=int, default=6970)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--max-ticks", type=int, default=5)
    parser.add_argument("--step-delay", type=float, default=0.5)
    parser.add_argument("--origin-lat", type=float, default=38.8895)
    parser.add_argument("--origin-lon", type=float, default=-77.0353)
    parser.add_argument("--altitude-ft", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=20.0)
    return parser.parse_args()


def start_tcp_listener(host: str, port: int, timeout: float, bucket: List[bytes]) -> threading.Thread:
    def _loop() -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            sock.listen(1)
            sock.settimeout(timeout)
            print(f"[listener] Waiting for TCP connection on {host}:{port}")
            try:
                conn, addr = sock.accept()
            except socket.timeout:
                print("[listener] Accept timed out")
                return
            print(f"[listener] Connection established from {addr}")
            with conn:
                conn.settimeout(0.5)
                deadline = time.time() + timeout
                while time.time() < deadline:
                    try:
                        data = conn.recv(65535)
                    except socket.timeout:
                        continue
                    if not data:
                        print("[listener] Remote closed connection")
                        break
                    bucket.append(data)
                    print(f"[listener] Received chunk ({len(data)} bytes)")
                print("[listener] Listener exiting")

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    return thread


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
        f"tcp://{args.cot_host}:{args.cot_port}",
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
    bucket: List[bytes] = []
    listener_thread = start_tcp_listener(args.cot_host, args.cot_port, args.timeout, bucket)
    try:
        run_cli(args)
    finally:
        listener_thread.join()
    total_bytes = sum(len(chunk) for chunk in bucket)
    print(f"Captured {len(bucket)} TCP chunks totaling {total_bytes} bytes.")


if __name__ == "__main__":
    main()
