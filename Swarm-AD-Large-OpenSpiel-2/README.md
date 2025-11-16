# Swarm-AD-Large-OpenSpiel-2

Version 2 of the high-fidelity Swarm Air Defense OpenSpiel environment. This drop mirrors the feature set of the original large game while registering a new `swarm_defense_large_v2` short name so it can coexist with the legacy build.

## Key differences vs. the abstract game

- **Drone saturation** – 40 simultaneous attackers with 20 interceptors (half of the attackers) and stochastic ToT windows.
- **Directional AD behavior** – Air-defense batteries track orientation, require finite rotation time, and possess limited fields of view.
- **Gridless arena** – Engagements unfold in a continuous 24×24 area. Drone paths are planned with RRT + shortcut smoothing and can leverage blueprint mid-waypoints.
- **Blueprint strategies** – Midpoint heuristics (`direct`, `fan_left/right`, `loiter`) steer RRT sampling bias to create richer trajectories.

## Modules

| File | Purpose |
| --- | --- |
| `swarm_defense_large_game.py` | Custom OpenSpiel game definition registered as `swarm_defense_large_v2`. |
| `pathfinding.py` | Lightweight RRT planner, smoothing, and sampling utilities for the gridless arena. |
| `blueprint_midpoints.py` | Stochastic two-midpoint generator that biases RRT sampling for each blueprint style. |
| `demo_visualizer.py` | Samples a fully large-game episode with random heuristics and renders a PNG snapshot. |
| `demo_animation.py` | Produces a time-stepped GIF showing drone motion and AD field-of-view rotations over the engagement. |

## Standalone demos

The revamped v2 build now operates independently of the abstract grid game. To
get a feel for the two-wave flow you can sample a full large-game episode with a
stochastic attacker/defender heuristic and render the outcome:

```bash
python Swarm-AD-Large-OpenSpiel-2/demo_visualizer.py --seed 1234
```

The script now routes through `demo_rollout.rollout_two_wave_episode`, which
forces realistic wave splitting. Wave 1 intentionally withholds part of the
drone inventory so Wave 2 always has units to launch once discoveries carry
over. The snapshot annotates the current phase, wave number, discovered AD halos,
entry markers per wave, and per-wave attrition stats so you can quickly see how
the engagement unfolded across both attack waves.

Need to inspect how the second wave behaves once AD sites are detected? Add
`--allow-early-ad-targets` to let drones strike AD units without waiting for
discovery, or stick with the default discovery requirement to mirror the new
game rules.

Need something animated? Produce a GIF that includes AD field-of-view slews and
interceptor launches:

```bash
python Swarm-AD-Large-OpenSpiel-2/demo_animation.py --seed 1337 --time-step 0.25 --fps 12
```

The animation CLI shares the same rollout helper and `--allow-early-ad-targets`
switch so both outputs remain consistent. Generated PNG/GIF assets land inside
the `Visualizer/` folder.
