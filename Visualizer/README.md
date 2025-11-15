# ESCHER Swarm Solver Visualizer

This package builds a lightweight OpenSpiel-inspired scenario simulator and publishes game state snapshots to WinTAK using [PyTak](https://github.com/snstac/pytak). It focuses on a 16×16 battlespace with attacking drones, interceptors, and air-defense (AD) assets. The generated Cursor-on-Target (CoT) events let you validate swarm behaviors without hosting a TAK server.

## Scenario Highlights

- **Grid:** 16×16, origin at the northwest corner.
- **Attackers:** 6 drones launched along the top row, each with a time-on-target (TOT) offset of +0 s, +2 s, or +4 s.
- **Interceptors:** 3 blue-force drones that prioritize high-value attackers based on assigned targets and TOT.
- **Air Defense:** 2 AD units positioned on an 8×8 stride lattice. Each has a 50% probabilistic kill if an attacker crosses its envelope.
- **Targets:** 3 value-tiered clusters seeded randomly across the bottom half of the grid.
- **Game Engine:** Deterministic movement with stochastic AD outcomes. The simulator produces a `GameHistory` compatible with OpenSpiel-style policies.
- **Kill Markers:** Destroyed attackers now retain their kill site, shooter, and intercept tick; CoT feeds emit dedicated kill icons so you can see exactly where interceptors and AD batteries scored hits.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
pip install -e .[dev]
swarm-visualizer --iterations 1 --cot-endpoint udp://127.0.0.1:6969 --dry-run
```

- Remove `--dry-run` to stream CoT to WinTAK directly (ensure your TAK client listens on the same UDP endpoint).
- Use `--export-file cot_log.xml` to persist the feed for later ingestion.
- `--cot-endpoint` must point to a TAK transport such as `udp://`, `udp+wo://`, `tcp://`, or `tls://`; HTTP(S) URLs are not valid for CoT streaming.
- Need a quick sanity check before running the full simulator? Use `python scripts/push_test_cot.py --cot-endpoint udp://127.0.0.1:6969 --uid TEST --callsign TEST` to fire a single CoT icon so you can confirm WinTAK is listening.
- Want to see multiple symbology examples at once? Run `python scripts/push_test_cot.py --cot-endpoint udp://127.0.0.1:6969 --mode sample-pack` to publish a curated set of hostile drones, blue-force interceptors, air-defense units, targets, and kill markers centered on the National Mall. Each icon prints its precise lat/lon as it transmits so you can pan the WinTAK map to the Washington Monument, White House, Capitol, Pentagon, and Joint Base Anacostia-Bolling with confidence.

## Project Layout

```
├── pyproject.toml
├── README.md
├── src
│   └── swarm_visualizer
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── cot.py
│       ├── entities.py
│       ├── history.py
│       ├── scenario.py
│       ├── simulator.py
│       └── pytak_client.py
└── tests
    └── test_scenario.py
```

## WinTAK Testbench Workflow

1. Launch WinTAK and ensure it is listening for UDP CoT traffic (Settings → Network → CoT → UDP Listener). Default: `udp://127.0.0.1:6969`.
2. Run `swarm-visualizer --iterations 5 --cot-endpoint udp://127.0.0.1:6969`.
3. Observe attacker, interceptor, and AD symbology appear on the WinTAK map. Each snapshot encodes the full board state with timestamps that honor the TOT offsets.
4. (Optional) Use the sample icon pack command above to keep a baseline set of representative symbols on the map while you iterate on TAK network settings.

## Advanced Options

- `--seed 13` – Reproduce a deterministic random layout.
- `--step-delay 1.0` – Control pacing (seconds) between snapshots.
- `--dry-run` – Print CoT XML payloads instead of sending them.
- `--export-file cot_log.xml` – Archive CoT stream for offline analysis.
- `--iterations 10` – Chain multiple simulated matches.
- `--origin-lat / --origin-lon / --altitude-ft` – Reposition the grid anywhere on earth. Defaults place the scenario over the Washington, DC National Mall so it is easy to spot from WinTAK.
    - If you are targeting a TAK server over TLS, use a `tls://host:port` URL. Do **not** supply `https://` or `http://`—those speak the MARTI REST API rather than the CoT transport and will be rejected by the CLI.
    - To validate network reachability independently of the simulator, leverage `scripts/push_test_cot.py` to drip one or more test events at your endpoint and watch them appear instantly in WinTAK.

### Washington, DC example (WSL → WinTAK)

```bash
wsl bash -lc 'cd "/mnt/c/Users/<you>/path/to/Visualizer" \
    && source .venv_wsl/bin/activate \
    && swarm-visualizer \
             --cot-endpoint udp+wo://172.17.128.1:6969 \
             --origin-lat 38.8895 \
             --origin-lon -77.0353 \
             --step-delay 0.75 \
             --iterations 1'
```

- Replace `172.17.128.1` with the Windows host IP shown by `ip route | grep default` inside WSL.
- Use the `udp+wo://` scheme when streaming from WSL so PyTak opens a write-only socket and does not attempt to bind the Windows address.
- If you prefer a different city, pass the appropriate latitude/longitude pair; all attackers, BLUFOR interceptors, AD units, and targets will render around that anchor point.

## Testing

Run the pytest suite to validate grid constraints, AD logic, and CoT serialization:

```bash
pytest
```

## License

MIT License © 2025 ESCHER Swarm Solver Team
