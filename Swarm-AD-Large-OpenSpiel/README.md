# Swarm-AD-Large-OpenSpiel

A high-fidelity, non-abstracted variant of the Swarm Air Defense game designed to stress-test blueprint strategies lifted from the abstract OpenSpiel environment.

## Key differences vs. the abstract game

- **Drone saturation** – 40 simultaneous attackers with 20 interceptors (half of the attackers) and stochastic ToT windows.
- **Directional AD behavior** – Air-defense batteries track orientation, require finite rotation time, and possess limited fields of view.
- **Gridless arena** – Engagements unfold in a continuous 24×24 area. Drone paths are planned with RRT + shortcut smoothing and can leverage blueprint mid-waypoints.
- **Blueprint strategies** – Midpoint heuristics (`direct`, `fan_left/right`, `loiter`) steer RRT sampling bias to create richer trajectories.

## Modules

| File | Purpose |
| --- | --- |
| `swarm_defense_large_game.py` | Custom OpenSpiel game definition registered as `swarm_defense_large`. |
| `pathfinding.py` | Lightweight RRT planner, smoothing, and sampling utilities for the gridless arena. |
| `blueprint_midpoints.py` | Stochastic two-midpoint generator that biases RRT sampling for each blueprint style. |
| `policy_transfer.py` | Transfers abstract Swarm-AD policies/states/actions into executable large-game blueprints. |
| `demo_visualizer.py` | Runs a random abstract episode, lifts it to a blueprint, executes it in the large game, and renders a PNG snapshot. |
| `demo_animation.py` | Produces a time-stepped GIF showing drone motion and AD field-of-view rotations over the engagement. |

## Using the transfer helpers

```python
import pyspiel
from ESCHER_Torch.eschersolver import ESCHERSolverTorch
from Swarm-AD-Large-OpenSpiel.policy_transfer import (
    lift_policy_to_blueprint,
    rollout_blueprint_episode,
)

# Assume ESCHER has been trained on the abstract "swarm_defense" game.
abstract_game = pyspiel.load_game("swarm_defense")
solver = ESCHERSolverTorch(abstract_game, num_iterations=50)
solver.iterate()  # or load checkpoints
blueprint = lift_policy_to_blueprint(solver.average_policy(), num_samples=6)
large_state = rollout_blueprint_episode(blueprint)
print("Blueprint damage:", large_state.returns()[0])
```

`lift_policy_to_blueprint` samples one or more abstract episodes, expands each drone assignment into four RRT-driven trajectories, and keeps the highest-priority 40 assignments (the attacker count in the large game). The helper preserves target/AD indices, remaps entry columns to continuous lanes, and selects stochastic dual midpoints per blueprint to diversify RRT exploration while still honoring the requested ToT.

## Demo visualizer

Generate a single shot demonstrating the lifted policy:

```bash
python Swarm-AD-Large-OpenSpiel/demo_visualizer.py --seed 1234
```

The script will save `Visualizer/swarm_defense_large_demo.png`, log attacker/defender payoffs, and show AD/interceptor kill tallies.

Generate an animation (GIF) that highlights AD field-of-view slews and drone timesteps:

```bash
python Swarm-AD-Large-OpenSpiel/demo_animation.py --seed 1337 --time-step 0.25 --fps 12
```

The animation is stored at `Visualizer/swarm_defense_large_animation.gif` by default.

## Notes for ESCHER integration

1. Train ESCHER on `swarm_defense` exactly as before (e.g., `python ESCHER-Torch/run_escher_torch_leduc.py` adapted to the Swarm game).
2. Grab the learned average policy via `solver.average_policy()`.
3. Call `lift_policy_to_blueprint(...)` to transform the abstract policy into a `BlueprintStrategy`.
4. Execute the blueprint inside `swarm_defense_large` either via `rollout_blueprint_episode` or by manually applying the encoded actions during the `SWARM_ASSIGNMENT` phase.
5. Optionally plug the resulting trajectories into the `Visualizer` or downstream simulators.

For advanced usage you can pass a defender policy into `rollout_blueprint_episode` (e.g., a heuristic intercept strategy) to analyze robustness.
