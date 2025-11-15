"""Command-line interface for the swarm visualizer."""

from __future__ import annotations

import argparse
from rich.console import Console

from .config import GeoConfig, GridConfig, PyTakRuntimeConfig, ScenarioBundle, ScenarioConfig
from .pytak_client import PyTakStreamer
from .scenario import ScenarioGenerator
from .simulator import SwarmGameSimulator

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Swarm OpenSpiel Visualizer")
    parser.add_argument("--iterations", type=int, default=1, help="How many matches to simulate")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed")
    parser.add_argument("--cot-endpoint", default="udp://127.0.0.1:6969", help="CoT endpoint URI")
    parser.add_argument("--callsign-prefix", default="SWARM", help="CoT UID prefix")
    parser.add_argument("--step-delay", type=float, default=1.0, help="Seconds between snapshots")
    parser.add_argument("--max-ticks", type=int, default=35, help="Maximum ticks per iteration")
    parser.add_argument("--dry-run", action="store_true", help="Print table instead of sending CoT")
    parser.add_argument("--export-file", default=None, help="Optional CoT XML output path")
    return parser.parse_args()


def build_bundle(seed: int | None) -> ScenarioBundle:
    scenario_cfg = ScenarioConfig(seed=seed)
    return ScenarioBundle(
        grid=GridConfig(),
        scenario=scenario_cfg,
        geo=GeoConfig(),
    )


def main() -> None:
    args = parse_args()
    bundle = build_bundle(args.seed)
    runtime = PyTakRuntimeConfig(
        cot_endpoint=args.cot_endpoint,
        cot_callsign_prefix=args.callsign_prefix,
        dry_run=args.dry_run,
        export_file=args.export_file,
        step_delay=args.step_delay,
    )

    console.print(
        f"[cyan]Running {args.iterations} iteration(s) with grid {bundle.grid.size}×{bundle.grid.size}"
    )

    for idx in range(args.iterations):
        console.print(f"[bold green]Iteration {idx+1}/{args.iterations}")
        generator = ScenarioGenerator(bundle)
        scenario = generator.generate()
        simulator = SwarmGameSimulator(scenario, bundle=bundle)
        history = simulator.run(max_ticks=args.max_ticks)
        streamer = PyTakStreamer(bundle=bundle, runtime=runtime)
        streamer.stream_sync(history)


if __name__ == "__main__":
    main()
