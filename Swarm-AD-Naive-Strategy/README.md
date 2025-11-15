# Swarm-AD Naive Strategy

This folder now hosts a deliberately simple blueprint runner **that is wired directly into the real `Swarm-AD-OpenSpiel` game**. Instead of approximating the engagement rules, we import the reference OpenSpiel game, execute a full episode, then render the results with extra annotations (interceptor lines, AD kill markers, destroyed targets, etc.). The default seed (503721862) reproduces the reference outcome with five total interceptions (2 AD + 3 interceptors).

## Requirements

- Python 3.10+ with `matplotlib` (install via `pip install -r requirements.txt`).
- A working OpenSpiel install (`pyspiel`). In this repository we run it via **WSL** because the Windows-native interpreter does not ship with `pyspiel`.
- Repository structure intact so the sibling `Swarm-AD-OpenSpiel/` folder is discoverable.

## Running the OpenSpiel-aligned demo

1. Open a WSL terminal in `ESCHER-Swarm-Solver/Swarm-AD-Naive-Strategy` (the WSL Python env must have `pyspiel`).
2. Install plotting deps if needed:

  ```bash
  pip install -r requirements.txt
  ```

3. Execute the runner (seed is optional; omit to use the default `503721862` baseline):

  ```bash
  python run_blueprint_demo.py --seed 503721862
  ```

4. Inspect the console summary plus the rendered PNG at `output/blueprint_demo.png`. The image highlights:
  - AD placement, coverage bubbles, and any ADs killed by drones.
  - Every drone path with ToT color-coding, including where early interceptions occur.
  - Interceptor engagements, AD intercept arcs, and final strike results against targets.

## Files

| File | Purpose |
| --- | --- |
| `blueprint_strategy.py` | Legacy blueprint helpers **plus** the new OpenSpiel bridge (`run_openspiel_episode`, `render_openspiel_episode`) that execute the real game and render annotated snapshots. |
| `run_blueprint_demo.py` | CLI wrapper that forwards the seed/output path, prints stats, and saves the visualization. |
| `requirements.txt` | Declares the visualization dependency (`matplotlib`). |

Feel free to swap in your own attacker/defender policies. As long as they produce an OpenSpiel `SwarmDefenseState`, the rendering helpers can visualize and summarize the engagement.
