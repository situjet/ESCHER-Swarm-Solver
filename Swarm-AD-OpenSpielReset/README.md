# Swarm Air-Defense Game

This folder contains a sequential, two-player OpenSpiel environment that captures
an attacker-defender interaction between a ten-drone swarm and a layered air
defense (two autonomous AD launchers plus three manned interceptors). The game
was designed to prototype the scenario described in the hackathon brief and to
serve as a lightweight testbed for the ESCHER solver.

## Game flow

1. **Chance** sequentially samples three unique cells on the bottom half of the
   16×16 grid (rows 8–15, no stride) and then shuffles the target values
   {10, 20, 40} across the sampled coordinates.
2. **Defender (player 1)** places two autonomous AD launchers anywhere on the
   bottom half as long as both row and column indices are even (stride-2 lattice).
   Each launcher automatically engages the earliest drone whose path crosses its
   5-cell circular radius; the kill chance follows `1 - exp(-r * L)` where `L`
   is the path length a drone flies inside the bubble and `r` defaults to 0.9.
3. **Attacker (player 0)** allocates ten drones sequentially. Every drone must
   launch from row 0 (any column), select either one of the three target clusters
   or one of the AD sites as its objective, and choose a time-on-target (ToT)
   offset from {0s, +2s, +4s}. The ToT acts as a delay before the drone departs
   row 0, so later choices literally arrive later in the engagement timeline.
4. **Defender (player 1)** receives the full attack plan and assigns three
   interceptors. Each interceptor can engage one surviving drone, with a “pass”
   option if fewer than three intercepts are desired. Interceptors are assumed
   to launch from row 15 and sprint up-range at twice the drone’s speed; if the
   geometry shows they cannot physically meet the drone before it impacts a
   target, the engagement is treated as a miss even if the action was selected.
5. **Chance** resolves autonomous AD shots via the probabilistic rule above. Drones that were tasked
   against an AD and survive the intercept+AD gauntlet will destroy that launcher
   on arrival, removing it from the fight. Drones that reach a target cluster
   contribute the cluster’s value as damage. The game is zero-sum: attacker payoff
   equals total damage; defender payoff is its negation.

The design intentionally avoids simulating continuous time—after decisions are
made, outcomes resolve immediately via deterministic logic plus the two chance
nodes (target sampling and AD kill probability).

## Files

| File | Purpose |
| --- | --- |
| `swarm_defense_game.py` | OpenSpiel game definition, registration hook, and helper snapshot utilities. |
| `demo_visualizer.py` | Plays a sample episode with simple heuristics and produces a Matplotlib snapshot stored under `Visualizer/`. |
| `__init__.py` | Re-export convenience for downstream imports. |
| `state_space_solver.py` | Generic OpenSpiel tree-expansion utility that reports node counts, branching factors, and runtime stats. |

## Quick start

```bash
python Swarm-AD-OpenSpiel/demo_visualizer.py
```

The script will:

1. Auto-register the `swarm_defense` game with OpenSpiel.
2. Roll out a complete episode using baseline heuristic policies.
3. Save a visual summary to `Visualizer/swarm_defense_demo.png` along with a
   short textual summary of the resulting payoffs.

Use `pyspiel.load_game("swarm_defense")` to plug the environment into ESCHER or
any other OpenSpiel solver. The `SwarmDefenseState.snapshot()` helper exposes a
structured view of the targets, AD unit status (alive/destroyed and by whom),
drone destinations, AD intercept footprints, and (when successful) the precise
coordinates and timestamps where interceptors killed a drone. The console
output now also reports how many drones were killed by ADs vs. interceptors vs.
those that slipped through, and the snapshot overlays the same breakdown.
AD kills now appear as bright yellow `X` markers connected back to the firing
launcher with a dotted tether and labeled `AD#→D#` time stamps, while
interceptor kills show up as cyan stars. If you do not see those overlays in
`Visualizer/swarm_defense_demo.png`, ensure you regenerated the image after
updating the repository.

### Tuning and diagnostics

- Set `SWARM_AD_KILL_RATE=<float>` before launching a simulation to bump the AD
   lethality up or down without editing code (default `0.9`).
- Run `python Swarm-AD-OpenSpiel/tools/kill_rate_report.py --episodes 200 --policy demo`
   from WSL to batch-roll episodes and print aggregate kill percentages for the
   demo heuristics (use `--policy random` to stress-test the model with random
   play). This utility mirrors the verification workflow used while calibrating
   the probabilistic AD system.

## State-space explorer

`state_space_solver.py` walks the underlying game tree so we can gauge how many
states ESCHER (or any other solver) will touch for a given configuration. It is
agnostic to the game definition, so as long as a game is registered with
OpenSpiel you can reuse the same CLI.

```bash
python Swarm-AD-OpenSpiel/state_space_solver.py \
   --game swarm_defense \
   --max-nodes 10000 \
   --progress 2
```

Key flags:

- `--traversal {dfs,bfs}`: choose depth-first (default) or breadth-first order.
- `--chance-mode {full,sample}`: enumerate every chance outcome or randomly
   sample one per chance node (useful when branching explodes).
- `--max-depth`, `--max-nodes`, `--max-terminals`, `--time-limit`: apply hard
   caps so experiments finish quickly when the state space is massive.
- `--progress <seconds>`: log periodic progress; set to `0` to silence updates.
- `--import-module <path>`: import extra Python modules before loading the game
   (handy when the game registers itself during import).

To enumerate every reachable state (useful for measuring the maximum search
space before launching ESCHER), run with exhaustive chance branching and unique
state tracking enabled:

```bash
python Swarm-AD-OpenSpiel/state_space_solver.py \
   --game swarm_defense \
   --chance-mode full \
   --count-unique \
   --progress 10
```

`--count-unique` hashes each state's action history so duplicates are skipped,
and the summary will report both the total number of unique states and the count
of terminal states. Remove depth/node caps to let the traversal finish only when
the entire tree has been explored.

The summary includes decision/chance/terminal node counts, branching-factor
stats, depth histogram, and per-phase visit counts (when the state exposes a
`phase()` method). Use these metrics to monitor how rule tweaks impact search
costs before running ESCHER sweeps.
