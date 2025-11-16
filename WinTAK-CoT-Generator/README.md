# WinTAK CoT Generator for Swarm Defense

This module generates Cursor-on-Target (CoT) sequences that visualize the Swarm Defense game on WinTAK.

## Overview

The generator creates time-stepped CoT events that mirror the visualization shown in the GIF animations, centered around Pittsburgh International Airport.

## Features

- Generates CoT sequences from Swarm Defense game states
- Reads recorded Swarm-AD-Large snapshots so the TAK feed exactly mirrors the MatPlotLib animation
- Supports both small and large game variants
- Pushes events to tcp://127.0.0.1:6969 for WinTAK visualization
- Geographic positioning centered on Pittsburgh International Airport (40.4915°N, 80.2329°W)

## Usage

```bash
python generate_cot_sequence.py --seed 42 --time-step 0.5 --interval 0.5 --fps-multiplier 2
```

### Copying the Swarm-AD-Large animation

1. Run the large-game animation exporter (this now emits `Visualizer/swarm_large_snapshot.json` by default):

	```bash
	python Swarm-AD-Large-OpenSpiel/demo_animation.py --seed 42 --time-step 0.25 --fps 12
	```

2. Launch the CoT generator. It automatically ingests the snapshot above so the TAK feed matches the MatPlotLib animation:

	```bash
	python WinTAK-CoT-Generator/generate_cot_sequence.py --seed 42 --interval 0.25
	```

Use `--snapshot /path/to/snapshot.json` to point at a different exported episode, or `--fresh-run` to ignore snapshots and synthesize a brand-new engagement.

### Key options

- `--fps-multiplier` – scales both the simulator step and CoT send interval. A value of 2 doubles the effective frame rate without changing your original CLI arguments.
- `--time-step` / `--interval` – base cadence before the FPS multiplier is applied.
- `--snapshot` – path to the Swarm-AD-Large snapshot JSON to replay (defaults to `Visualizer/swarm_large_snapshot.json`).
- `--fresh-run` – skip snapshot replay and generate a brand-new scenario.
- `--dry-run` – print events instead of sending them to WinTAK for quick inspection.

## Files

- `generate_cot_sequence.py` - Main generator script
- `cot_helpers.py` - CoT formatting utilities
- `game_simulator.py` - Game state simulation logic
