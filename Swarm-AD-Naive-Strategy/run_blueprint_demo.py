from __future__ import annotations

import argparse
from pathlib import Path

from blueprint_strategy import (
    DEFAULT_BASELINE_SEED,
    format_episode_summary,
    render_openspiel_episode,
    run_openspiel_episode,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the OpenSpiel-aligned Swarm AD blueprint demo")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_BASELINE_SEED,
        help="Optional RNG seed forwarded to the OpenSpiel episode runner",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the rendered snapshot PNG",
    )
    args = parser.parse_args()

    episode = run_openspiel_episode(args.seed)
    output_path = args.output or (Path(__file__).parent / "output" / "blueprint_demo.png")

    print("=== OpenSpiel Episode Summary ===")
    for line in format_episode_summary(episode):
        print(f"  - {line}")
    saved_path = render_openspiel_episode(episode, output_path)
    print(f"Snapshot written to {saved_path}")


if __name__ == "__main__":
    main()
