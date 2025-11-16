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

### Replaying ESCHER-Torch inference runs

1. Run the policy inference helper. Each episode directory now contains a WinTAK-ready JSON snapshot alongside the GIF/PNG artifacts (disable via `--no-wintak-snapshot` if you do not need it):

	```bash
	python ESCHER-Torch/run_swarm_large_inference.py \
	  --checkpoint ESCHER-Torch/results/swarm_defense_large_v2/2025_11_15_19_06_19 \
	  --scenario-mode single --episodes 1 --seed 4 --no-animation --no-snapshot
	```

2. Point the generator at the resulting `episode_summary.json`. The script resolves the `wintak_snapshot.json` referenced inside the summary and replays it frame-for-frame on TAK:

	```bash
	python WinTAK-CoT-Generator/generate_cot_sequence.py \
	  --summary ESCHER-Torch/results/swarm_defense_large_v2/inference_runs/episode_01_seed_4/episode_summary.json \
	  --dry-run
	```

This workflow guarantees your TAK feed is a perfect replica of the policy rollout captured during inference, without re-simulating any randomness.

### Key options

- `--fps-multiplier` – scales both the simulator step and CoT send interval. A value of 2 doubles the effective frame rate without changing your original CLI arguments.
- `--time-step` / `--interval` – base cadence before the FPS multiplier is applied.
- `--snapshot` – path to the Swarm-AD-Large snapshot JSON to replay (defaults to `Visualizer/swarm_large_snapshot.json`).
- `--summary` – path to an `episode_summary.json` emitted by `run_swarm_large_inference.py`; automatically loads the referenced snapshot.
- `--fresh-run` – skip snapshot replay and generate a brand-new scenario.
- `--dry-run` – print events instead of sending them to WinTAK for quick inspection.

## Files

- `generate_cot_sequence.py` - Main generator script
- `cot_helpers.py` - CoT formatting utilities
- `game_simulator.py` - Game state simulation logic
