"""Render an existing Swarm Defense large snapshot into an animation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from demo_animation import _build_animation  # type: ignore


def _load_snapshot(snapshot_path: Path) -> Dict[str, Any]:
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

    with snapshot_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("Snapshot JSON must be an object")

    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a precomputed Swarm Defense large snapshot JSON and build a "
            "Matplotlib animation matching the WinTAK export."
        )
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        required=True,
        help="Path to wintak_snapshot.json produced by run_swarm_large_inference.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output GIF path (defaults to <snapshot stem>.gif)",
    )
    parser.add_argument(
        "--time-step",
        type=float,
        default=0.25,
        help="Simulation timestep in seconds",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=12,
        help="Frames per second for the generated animation",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    snapshot_path = args.snapshot.expanduser()
    output_path = args.output.expanduser() if args.output else snapshot_path.with_suffix(".gif")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    snapshot = _load_snapshot(snapshot_path)
    _build_animation(snapshot, output_path, time_step=args.time_step, fps=args.fps)

    print(f"Seed: {snapshot.get('seed', 'unknown')}")
    print(f"Snapshot: {snapshot_path}")
    print(f"Animation: {output_path}")


if __name__ == "__main__":
    main()
