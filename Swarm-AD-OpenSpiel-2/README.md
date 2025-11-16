# 2Swarm2

2Swarm2 is a two-wave extension of the original Swarm-AD OpenSpiel environments. It retains the layered air-defense mechanics while introducing:

- **Two attack waves:** Wave one allows the attacker to probe with any subset of drones. Wave two automatically deploys the remaining drones with the additional intel gathered.
- **AD discovery:** Any drone that enters an AD bubble reveals that defense site for the following wave, unlocks direct strikes against it, and informs survivor planning.
- **Shared uncertainty:** Attackers and defenders cannot observe each other’s remaining ammunition, forcing strategic wave splits and intercept rationing.
- **Interceptor economy:** Any unused interceptor ammunition at the end of wave two is converted into a defender bonus, reinforcing ammo conservation.

The implementation lives entirely under `Swarm-AD-OpenSpiel-2/` so it can evolve independently of the legacy games.

## Running the tests

The tests rely on OpenSpiel (`pyspiel`). From your WSL bash shell:

```bash
cd /c/Users/situj/Desktop/Carnegie\ Mellon/Hackathon/ESCHER-Swarm-Solver
wsl python3 -m pytest Swarm-AD-OpenSpiel-2/tests
```

> ℹ️ Always prefix commands with `wsl python3 ...` (not `wsl.exe && python3 ...`).
> The interpreter and `pyspiel` live inside the WSL environment, so both the
> `wsl` shim and `python3` must run in the same invocation.

The suite validates the wave gating logic and the interceptor bonus payout.

## Rendering a GIF demo

A lightweight matplotlib visualizer is included. It rolls out a seeded episode with a simple heuristic policy and exports an animated GIF.

```bash
cd /c/Users/situj/Desktop/Carnegie\ Mellon/Hackathon/ESCHER-Swarm-Solver
wsl python3 Swarm-AD-OpenSpiel-2/visualize_2swarm2.py --seed 11 --output Swarm-AD-OpenSpiel-2/output/2swarm2_demo.gif
```

Dependencies: `pyspiel`, `matplotlib`, and `pillow`. The script automatically creates the `output/` directory and reports the saved path. The renderer now:

- staggers the two attack waves in time to make the hand-off explicit,
- draws interceptor launch traces and kill markers, and
- color-codes kill events (AD, interceptor, successful strikes) for quick review.

## Module layout

```
Swarm-AD-OpenSpiel-2/
├─ README.md                  # This guide
├─ __init__.py               # Thin re-export for convenience
├─ Swarm_AD_OpenSpiel_2/
│  ├─ __init__.py           # Namespace-friendly package
│  └─ two_swarm2_game.py    # Full OpenSpiel implementation
├─ tests/
│  └─ test_two_swarm2_game.py
├─ visualize_2swarm2.py      # GIF generator / demo runner
└─ todo.txt                  # Original requirements reference
```

You can import the game via:

```python
from Swarm_AD_OpenSpiel_2 import TwoSwarm2Game
```

OpenSpiel will also register the environment under the short name `two_swarm2`, enabling use from generic tooling (e.g., `pyspiel.load_game("two_swarm2")`).
