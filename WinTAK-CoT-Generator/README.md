# WinTAK CoT Generator for Swarm Defense

This module generates Cursor-on-Target (CoT) sequences that visualize the Swarm Defense game on WinTAK.

## Overview

The generator creates time-stepped CoT events that mirror the visualization shown in the GIF animations, centered around Pittsburgh International Airport.

## Features

- Generates CoT sequences from Swarm Defense game states
- Supports both small and large game variants
- Pushes events to tcp://127.0.0.1:6969 for WinTAK visualization
- Geographic positioning centered on Pittsburgh International Airport (40.4915°N, 80.2329°W)

## Usage

```bash
python generate_cot_sequence.py --seed 42 --time-step 0.5 --interval 0.5
```

## Files

- `generate_cot_sequence.py` - Main generator script
- `cot_helpers.py` - CoT formatting utilities
- `game_simulator.py` - Game state simulation logic
