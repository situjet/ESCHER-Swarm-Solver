"""Demo runner and visualizer for the Swarm Defense OpenSpiel game."""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
from matplotlib import animation, patches
import numpy as np
import pyspiel
import torch

# Try to import yaml for config file support
try:
    import yaml
except ImportError:
    yaml = None

# Add ESCHER-Torch to path for policy loading
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "ESCHER-Torch"))

try:
    from ESCHER_Torch.eschersolver import PolicyNetwork
    POLICY_AVAILABLE = True
except ImportError:
    POLICY_AVAILABLE = False
    PolicyNetwork = None

from swarm_defense_game import (
    Phase,
    SwarmDefenseState,
    TOT_CHOICES,
)

# These will be set dynamically based on game config
GRID_SIZE = 16
BOTTOM_HALF_START = GRID_SIZE // 2
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "Visualizer"
OUTPUT_PATH = OUTPUT_DIR / "swarm_defense_demo.png"
ANIMATION_OUTPUT_PATH = OUTPUT_DIR / "swarm_defense_animation.gif"

TOT_PALETTE = {
    TOT_CHOICES[0]: "tab:red",
    TOT_CHOICES[1]: "tab:orange",
    TOT_CHOICES[2]: "tab:purple",
}

AD_KILL_COLOR = "#FF3B30"
AD_KILL_EDGE = "black"
AD_KILL_LINK = "#8C1B13"
INTERCEPTOR_KILL_COLOR = "tab:cyan"
TARGET_KILL_COLOR = "#F1C40F"
TARGET_KILL_EDGE = "#7D6608"
TARGET_KILL_MARKER = "P"
DRONE_SPEED = 1.0  # Grid cells per time unit


def _compute_target_kill_status(
    drones: Tuple[Dict[str, object], ...], targets: Tuple[object, ...]
) -> Tuple[Dict[str, object], ...]:
    statuses = []
    for _ in targets:
        statuses.append({
            "destroyed": False,
            "time": None,
            "drone": None,
            "damage": 0.0,
        })

    for idx, drone in enumerate(drones):
        target_idx = drone.get("target_idx")
        if target_idx is None or target_idx >= len(statuses):
            continue
        if not drone.get("strike_success"):
            continue
        entry_row, entry_col = drone["entry"]
        dest_row, dest_col = drone["destination"]
        tot_value = float(drone.get("tot", 0.0))
        arrival_time = tot_value + math.dist((entry_row, entry_col), (dest_row, dest_col))
        status = statuses[target_idx]
        if (not status["destroyed"]) or (arrival_time < status["time"]):
            status.update(
                destroyed=True,
                time=arrival_time,
                drone=idx,
                damage=float(drone.get("damage_inflicted") or 0.0),
            )
    return tuple(statuses)


def _count_outcomes(drones: Tuple[Dict[str, object], ...]) -> Tuple[int, int, int, int]:
    ad = inter = surv = ad_target = 0
    for drone in drones:
        destroyed_by = drone.get("destroyed_by") or ""
        if isinstance(destroyed_by, str) and destroyed_by.startswith("ad"):
            if destroyed_by.startswith("ad:"):
                ad += 1
            else:
                ad_target += 1
        elif isinstance(destroyed_by, str) and destroyed_by.startswith("interceptor"):
            inter += 1
        else:
            surv += 1
    return ad, inter, surv, ad_target


def _sample_chance_action(state: SwarmDefenseState, rng: random.Random) -> int:
    outcomes = state.chance_outcomes()
    pick = rng.random()
    cumulative = 0.0
    for action, prob in outcomes:
        cumulative += prob
        if pick <= cumulative:
            return action
    return outcomes[-1][0]


def _parse_coords_from_label(label: str) -> Tuple[int, int]:
    _, coords = label.split(":", maxsplit=1)
    coords = coords.strip()
    coords = coords.strip("()")
    row_str, col_str = coords.split(",")
    return int(row_str), int(col_str)


def _defender_ad_policy(state: SwarmDefenseState, rng: random.Random) -> int:
    legal = state.legal_actions()
    snapshot = state.snapshot()
    targets: Tuple[Dict[str, object], ...] = snapshot.get("targets", ())
    player = state.current_player()

    def score(action: int) -> float:
        label = state.action_to_string(player, action)
        row, col = _parse_coords_from_label(label)
        total = 0.0
        for target in targets:
            t_row = target.row  # type: ignore[attr-defined]
            t_col = target.col  # type: ignore[attr-defined]
            dist = abs(t_row - row) + abs(t_col - col)
            weight = getattr(target, "value", 0.0)
            total -= dist * 0.1
            total += weight * 0.01
        return total + rng.random() * 0.01

    best_action = max(legal, key=score)
    return best_action


def _defender_interceptor_policy(state: SwarmDefenseState, rng: random.Random) -> int:
    legal = state.legal_actions()
    snapshot = state.snapshot()
    drones: Tuple[Dict[str, object], ...] = snapshot.get("drones", ())
    current_player = state.current_player()
    best_action = legal[-1]
    best_score = float("-inf")
    for action in legal:
        label = state.action_to_string(current_player, action)
        if not label.startswith("interceptor:drone"):
            continue
        idx = int(label.split("=")[-1])
        if idx >= len(drones):
            continue
        drone_info = drones[idx]
        if drone_info["destroyed_by"] is not None:
            continue
        target_value = drone_info.get("target_value") or 0.0
        tot_value = drone_info["tot"]
        score = target_value * 10 - tot_value + rng.random()
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


def _defender_policy(state: SwarmDefenseState, rng: random.Random) -> int:
    if state.phase() == Phase.AD_PLACEMENT:
        return _defender_ad_policy(state, rng)
    if state.phase() == Phase.INTERCEPT_ASSIGNMENT:
        return _defender_interceptor_policy(state, rng)
    return random.choice(state.legal_actions())


def _attacker_policy(state: SwarmDefenseState, rng: random.Random) -> int:
    return rng.choice(state.legal_actions())


class PolicyWrapper:
    """Wrapper to use trained policy network for action selection."""
    
    def __init__(
        self,
        policy_network: PolicyNetwork,
        player_id: int,
        device: str = "cpu",
    ):
        self.policy_network = policy_network
        self.player_id = player_id
        self.device = device
        self.policy_network.eval()
        self.policy_network.to(device)
    
    def get_action(self, state: SwarmDefenseState, rng: random.Random, use_sampling: bool = False) -> int:
        """Get action from the trained policy.
        
        Args:
            state: Current game state
            rng: Random number generator for sampling
            use_sampling: If True, sample from distribution; if False, take argmax
        """
        cur_player = state.current_player()
        if cur_player != self.player_id:
            raise ValueError(f"Policy is for player {self.player_id} but state is for player {cur_player}")
        
        legal_actions = state.legal_actions(cur_player)
        if len(legal_actions) == 1:
            return legal_actions[0]
        
        # Get info state and mask
        info_state = np.array(state.information_state_tensor(cur_player), dtype=np.float32)
        mask = np.array(state.legal_actions_mask(cur_player), dtype=np.float32)
        
        # Forward pass through network
        with torch.no_grad():
            info_tensor = torch.from_numpy(info_state).unsqueeze(0).to(self.device)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(self.device)
            probs = self.policy_network(info_tensor, mask_tensor)[0].cpu().numpy()
        
        # Sample or take argmax
        if use_sampling:
            # Sample according to probabilities
            action = rng.choices(legal_actions, weights=[probs[a] for a in legal_actions])[0]
        else:
            # Take the action with highest probability
            action = max(legal_actions, key=lambda a: probs[a])
        
        return action


def show_state_vector_breakdown(state: SwarmDefenseState, step_name: str = "") -> None:
    """Show the parts of the state vector that should change.
    
    Args:
        state: The game state to analyze
        step_name: Optional label for this breakdown
    """
    import numpy as np
    from swarm_defense_game import TOT_CHOICES
    
    vec = np.array(state.information_state_tensor(0))
    
    # Phase encoding (first 10 values - one-hot for 10 phases)
    phase_idx = np.where(vec[:10] > 0.5)[0]
    phase_name = state.phase().name
    
    num_targets = state.config.num_targets
    num_ad_units = state.config.num_ad_units
    num_drones = state.config.num_attacking_drones
    
    # Calculate indices based on structure:
    # Phase: 0-9 (10 values for 10 phases)
    # Target positions: 10 to 10+num_targets*2 (row, col pairs)
    # Target values: 10+num_targets*2 to 10+num_targets*3
    # AD units: 10+num_targets*3 to 10+num_targets*3+num_ad_units*3 (row, col, alive)
    # Interceptor info: 1 value (remaining interceptors)
    # Drone plans: num_drones * 6 (entry_col, target_idx, tot_idx, alive, interceptor_assigned, destroyed_by_interceptor)
    
    phase_start, phase_end = 0, 10
    target_pos_start = 10
    target_pos_end = target_pos_start + num_targets * 2
    target_val_start = target_pos_end
    target_val_end = target_val_start + num_targets
    ad_start = target_val_end
    ad_end = ad_start + num_ad_units * 3
    interceptor_start = ad_end
    interceptor_end = interceptor_start + 1
    drone_start = interceptor_end
    drone_end = drone_start + num_drones * 6
    
    print(f"\n  [{step_name}] State Vector Breakdown:")
    print(f"    Vector length: {len(vec)}")
    print(f"    Phase (indices {phase_start}-{phase_end-1}): {vec[phase_start:phase_end]} -> {phase_name}")
    print(f"    Target positions (indices {target_pos_start}-{target_pos_end-1}):")
    for i in range(num_targets):
        idx = target_pos_start + i * 2
        row_val = vec[idx]
        col_val = vec[idx + 1]
        row = int(row_val * state.config.grid_rows) if row_val > 0 else None
        col = int(col_val * state.config.grid_cols) if col_val > 0 else None
        print(f"      Target {i}: row={row_val:.3f} ({row}), col={col_val:.3f} ({col})")
    print(f"    Target values (indices {target_val_start}-{target_val_end-1}): {vec[target_val_start:target_val_end]}")
    print(f"    AD units (indices {ad_start}-{ad_end-1}):")
    for i in range(num_ad_units):
        idx = ad_start + i * 3
        row_val = vec[idx]
        col_val = vec[idx + 1]
        alive_val = vec[idx + 2]
        row = int(row_val * state.config.grid_rows) if row_val > 0 else None
        col = int(col_val * state.config.grid_cols) if col_val > 0 else None
        print(f"      AD {i}: row={row_val:.3f} ({row}), col={col_val:.3f} ({col}), alive={alive_val:.1f}")
    
    # Interceptor info (only if vector is long enough)
    if interceptor_start < len(vec):
        print(f"    Interceptor info (indices {interceptor_start}-{min(interceptor_end-1, len(vec)-1)}): {vec[interceptor_start:min(interceptor_end, len(vec))]}")
    else:
        print(f"    Interceptor info: Not yet in vector (vector length: {len(vec)})")
    
    # Drone plans (only show what fits in the vector)
    print(f"    Drone plans (indices {drone_start}-{min(drone_end-1, len(vec)-1)}):")
    vec_len = len(vec)
    max_drones_to_show = min(num_drones, (vec_len - drone_start) // 6) if vec_len > drone_start else 0
    
    for i in range(max_drones_to_show):
        idx = drone_start + i * 6
        if idx + 5 < vec_len:
            entry_col_val = vec[idx]
            target_idx_val = vec[idx + 1]
            tot_idx_val = vec[idx + 2]
            alive_val = vec[idx + 3]
            interceptor_assigned_val = vec[idx + 4]
            destroyed_by_interceptor_val = vec[idx + 5]
            
            entry_col = int(entry_col_val * state.config.grid_cols) if entry_col_val > 0 else None
            target_idx = int(target_idx_val * state.config.drone_target_slots) if target_idx_val > 0 else None
            tot_idx = int(tot_idx_val * len(TOT_CHOICES)) if tot_idx_val > 0 else None
            tot = TOT_CHOICES[tot_idx] if tot_idx is not None and tot_idx < len(TOT_CHOICES) else None
            
            # Determine target type
            if target_idx is not None:
                if target_idx < num_targets:
                    target_type = f"target:{target_idx}"
                else:
                    ad_idx = target_idx - num_targets
                    target_type = f"ad:{ad_idx}"
            else:
                target_type = "None"
            
            print(f"      Drone {i}: entry_col={entry_col_val:.3f} ({entry_col}), "
                  f"target_idx={target_idx_val:.3f} ({target_type}), "
                  f"tot_idx={tot_idx_val:.3f} ({tot_idx}, ToT={tot}), "
                  f"alive={alive_val:.1f}, interceptor_assigned={interceptor_assigned_val:.1f}, "
                  f"destroyed_by_interceptor={destroyed_by_interceptor_val:.1f}")
    
    if max_drones_to_show < num_drones:
        print(f"      ... ({num_drones - max_drones_to_show} more drone slots not yet in vector)")
    
    # Show actual state data if available
    if hasattr(state, '_target_positions') and len(state._target_positions) > 0:
        print(f"    Target positions (in _target_positions): {state._target_positions}")
    if hasattr(state, '_targets') and len(state._targets) > 0:
        print(f"    Actual targets (in _targets): {[(t.row, t.col, t.value) for t in state._targets]}")
    if hasattr(state, '_ad_units') and len(state._ad_units) > 0:
        print(f"    Actual AD units (in _ad_units): {[(u.row, u.col, u.alive) for u in state._ad_units]}")
    if hasattr(state, '_drone_plans') and len(state._drone_plans) > 0:
        print(f"    Actual drone plans (in _drone_plans):")
        for i, plan in enumerate(state._drone_plans):
            target_type = f"target:{plan.target_idx}" if plan.target_idx < num_targets else f"ad:{plan.target_idx - num_targets}"
            print(f"      Drone {i}: entry=(0,{plan.entry_col}) -> {target_type}, ToT={TOT_CHOICES[plan.tot_idx]}, destroyed_by={plan.destroyed_by}")


def load_policy(
    policy_path: Path,
    game: pyspiel.Game,
    device: str = "cpu",
    policy_layers: Optional[Tuple[int, ...]] = None,
) -> PolicyNetwork:
    """Load a trained policy network from a checkpoint.
    
    Args:
        policy_path: Path to the .pt file containing policy weights
        game: The OpenSpiel game (needed to get state tensor size)
        device: Device to load model on
        policy_layers: Architecture of the policy network (must match training).
                       If None, defaults to (256, 128) which is the standard ESCHER architecture.
    
    Returns:
        Loaded PolicyNetwork
    """
    if not POLICY_AVAILABLE:
        raise RuntimeError("PolicyNetwork not available. Make sure ESCHER-Torch is installed.")
    
    # Use default architecture if not specified
    if policy_layers is None:
        policy_layers = (256, 128)  # Standard ESCHER architecture
    
    # Get dimensions from game
    initial_state = game.new_initial_state()
    input_size = len(initial_state.information_state_tensor(0))
    num_actions = game.num_distinct_actions()
    
    # Create network with same architecture as training
    policy_net = PolicyNetwork(
        input_size=input_size,
        hidden_layers=policy_layers,
        num_actions=num_actions,
    )
    
    # Load weights
    state_dict = torch.load(str(policy_path), map_location=device)
    policy_net.load_state_dict(state_dict)
    policy_net.eval()
    
    return policy_net


def play_episode(
    seed: Optional[int] = None,
    game_params: Optional[Dict] = None,
    policy_path_p0: Optional[Path] = None,
    policy_path_p1: Optional[Path] = None,
    use_sampling: bool = False,
    device: str = "cpu",
    policy_layers: Optional[Tuple[int, ...]] = None,
) -> Tuple[SwarmDefenseState, int]:
    """Play a game episode.
    
    Args:
        seed: Optional RNG seed for reproducibility
        game_params: Optional dict of game parameters to pass to pyspiel.load_game()
        policy_path_p0: Optional path to policy.pt file for player 0 (attacker)
        policy_path_p1: Optional path to policy.pt file for player 1 (defender)
        use_sampling: If True, sample from policy; if False, take argmax
        device: Device for policy network inference
        policy_layers: Optional policy network architecture (defaults to (256, 128))
    """
    if seed is None:
        seed = random.SystemRandom().randint(0, 2**31 - 1)
    rng = random.Random(seed)
    import swarm_defense_game  # noqa: F401  # Registers the custom game.

    if game_params:
        # Format as spec string
        param_parts = [f"{key}={value}" for key, value in game_params.items()]
        spec_string = f"swarm_defense({','.join(param_parts)})"
        print(f"Loading game with parameters: {spec_string}")
        game = pyspiel.load_game(spec_string)
    else:
        print("Loading game with default parameters")
        game = pyspiel.load_game("swarm_defense")
    
    # Verify game configuration
    if hasattr(game, 'config'):
        print(f"Game configuration:")
        print(f"  - Grid: {game.config.grid_rows}x{game.config.grid_cols}")
        print(f"  - Targets: {game.config.num_targets}")
        print(f"  - AD units: {game.config.num_ad_units}")
        print(f"  - Attacking drones: {game.config.num_attacking_drones}")
        print(f"  - Interceptors: {game.config.num_interceptors}")
    
    state = game.new_initial_state()
    
    # Load policies for both players if provided
    policy_wrapper_p0 = None
    policy_wrapper_p1 = None
    
    if policy_path_p0 is not None:
        print(f"Loading Player 0 (Attacker) policy from: {policy_path_p0}")
        policy_net_p0 = load_policy(policy_path_p0, game, device=device, policy_layers=policy_layers)
        policy_wrapper_p0 = PolicyWrapper(policy_net_p0, 0, device=device)
        print(f"✓ Player 0 policy loaded")
    
    if policy_path_p1 is not None:
        print(f"Loading Player 1 (Defender) policy from: {policy_path_p1}")
        policy_net_p1 = load_policy(policy_path_p1, game, device=device, policy_layers=policy_layers)
        policy_wrapper_p1 = PolicyWrapper(policy_net_p1, 1, device=device)
        print(f"✓ Player 1 policy loaded")
    
    step = 0
    while not state.is_terminal():
        player = state.current_player()
        phase = state.phase() if hasattr(state, 'phase') else None
        
        # Debug: Track drone assignments
        if phase == Phase.SWARM_ASSIGNMENT:
            num_drones_assigned = len(state._drone_plans) if hasattr(state, '_drone_plans') else 0
            legal_actions = state.legal_actions(player) if player != pyspiel.PlayerId.CHANCE else []
            print(f"  Step {step}: Phase={phase.name}, Player={player}, Drones assigned: {num_drones_assigned}/{game.config.num_attacking_drones}, Legal actions: {len(legal_actions)}")
            if num_drones_assigned >= game.config.num_attacking_drones:
                print(f"    WARNING: All drones should be assigned, but phase is still SWARM_ASSIGNMENT!")
        
        try:
            if player == pyspiel.PlayerId.CHANCE:
                action = _sample_chance_action(state, rng)
            elif player == 0:
                if policy_wrapper_p0 is not None:
                    # Use trained policy for attacker
                    action = policy_wrapper_p0.get_action(state, rng, use_sampling=use_sampling)
                else:
                    # Use default attacker policy
                    action = _attacker_policy(state, rng)
            else:  # player == 1
                if policy_wrapper_p1 is not None:
                    # Use trained policy for defender
                    action = policy_wrapper_p1.get_action(state, rng, use_sampling=use_sampling)
                else:
                    # Use default defender policy
                    action = _defender_policy(state, rng)
            
            # Verify action is legal
            legal_actions = state.legal_actions(player) if player != pyspiel.PlayerId.CHANCE else [a for a, _ in state.chance_outcomes()]
            if action not in legal_actions:
                print(f"  ERROR: Action {action} is not legal! Legal actions: {legal_actions[:10]}...")
                # Fallback to first legal action
                action = legal_actions[0] if legal_actions else action
            
            state.apply_action(action)
        except Exception as e:
            print(f"  ERROR at step {step}: {e}")
            print(f"    Phase: {phase.name if phase else 'unknown'}")
            print(f"    Player: {player}")
            if hasattr(state, '_drone_plans'):
                print(f"    Drones assigned: {len(state._drone_plans)}/{game.config.num_attacking_drones}")
            raise
        step += 1
    
    # Final check
    snapshot = state.snapshot()
    num_drones_final = len(snapshot.get("drones", []))
    print(f"\n✓ Episode complete after {step} steps")
    print(f"  Final drones in snapshot: {num_drones_final} (expected: {game.config.num_attacking_drones})")
    
    return state, seed


def render_snapshot(state: SwarmDefenseState, output_path: Path) -> None:
    snapshot = state.snapshot()
    targets = snapshot["targets"]
    drones = snapshot["drones"]
    ad_units = snapshot["ad_units"]
    target_statuses = _compute_target_kill_status(drones, targets)
    ad_kills, interceptor_kills, survivors, ad_attrit = _count_outcomes(drones)

    # Get grid size from game config
    config = state.config
    grid_rows = config.grid_rows
    grid_cols = config.grid_cols
    bottom_half_start = config.bottom_half_start
    
    # Get AD position candidates from config
    ad_position_candidates = config.get_ad_position_candidates()

    fig, ax = plt.subplots(figsize=(10, 10))
    grid = np.zeros((grid_rows, grid_cols))
    ax.imshow(grid, cmap="Greys", alpha=0.05, extent=(-0.5, grid_cols - 0.5, grid_rows - 0.5, -0.5))
    ax.set_xticks(range(grid_cols))
    ax.set_yticks(range(grid_rows))
    ax.grid(True, which="both", color="lightgrey", linewidth=0.5, alpha=0.5)
    ax.set_xlim(-0.5, grid_cols - 0.5)
    ax.set_ylim(grid_rows - 0.5, -0.5)
    ax.set_title("Swarm Defense Episode Snapshot")

    ax.add_patch(
        plt.Rectangle(
            (-0.5, bottom_half_start - 0.5),
            grid_cols,
            grid_rows - bottom_half_start,
            linewidth=1.0,
            edgecolor="tab:blue",
            facecolor="none",
            linestyle="--",
            label="Bottom-half AO",
        )
    )
    # highlight even-stride candidate cells
    for row, col in ad_position_candidates:
        ax.scatter(col, row, s=10, color="tab:blue", alpha=0.3)

    destroyed_target_label_added = False
    for idx, target in enumerate(targets):
        status = target_statuses[idx]
        destroyed = status["destroyed"]
        if destroyed:
            label = None
            if not destroyed_target_label_added:
                label = "Destroyed target"
                destroyed_target_label_added = True
            time_str = f"{status['time']:.1f}" if status["time"] is not None else "?"
            ax.scatter(
                target.col,
                target.row,
                s=260,
                marker=TARGET_KILL_MARKER,
                color=TARGET_KILL_COLOR,
                edgecolors=TARGET_KILL_EDGE,
                linewidths=1.8,
                zorder=6,
                label=label,
            )
            caption = f"T{idx}\nV={target.value}\nD{status['drone']} t={time_str}"
        else:
            ax.scatter(target.col, target.row, s=200, color="tab:green", marker="o", zorder=4)
            caption = f"T{idx}\nV={target.value}"
        ax.text(
            target.col + 0.2,
            target.row + 0.2,
            caption,
            color="black",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none"} if destroyed else None,
        )

    ad_positions: Dict[int, Tuple[float, float]] = {}
    for idx, unit in enumerate(ad_units):
        col, row = unit["position"][1], unit["position"][0]
        ad_positions[idx] = (col, row)
        alive = bool(unit["alive"])
        color = "tab:blue" if alive else "tab:gray"
        marker = "^" if alive else "v"
        ax.scatter(col, row, marker=marker, s=200, color=color)
        status = "alive" if alive else f"KO ({unit.get('destroyed_by') or 'drone'})"
        ax.text(col - 0.6, row - 0.4, f"AD{idx}\n{status}", color=color, fontsize=8)

    ad_kill_label_added = False
    interceptor_kill_label_added = False
    for idx, drone in enumerate(drones):
        entry_col, entry_row = drone["entry"][1], drone["entry"][0]
        tgt_row, tgt_col = drone["destination"]
        tot = drone["tot"]
        color = TOT_PALETTE[tot]
        linestyle = "-" if drone["destroyed_by"] is None else "--"
        ax.plot([entry_col, tgt_col], [entry_row, tgt_row], color=color, linestyle=linestyle, linewidth=2)
        ax.scatter(entry_col, entry_row, color=color, marker="s", s=60)
        marker = "o"
        destroyed_by = drone["destroyed_by"] or ""
        if destroyed_by.startswith("interceptor"):
            marker = "x"
        elif destroyed_by.startswith("ad"):
            marker = "D"
        ax.scatter(tgt_col, tgt_row, color=color, marker=marker, s=120, facecolors="none")
        ax.text(
            tgt_col + 0.1,
            tgt_row - 0.2,
            f"D{idx}\nToT={tot}",
            color=color,
        )
        for ad_idx, intercept_point, intercept_time in drone["intercepts"]:
            hit_row, hit_col = intercept_point
            label = None
            if not ad_kill_label_added:
                label = "AD kill"
                ad_kill_label_added = True
            ad_col, ad_row = ad_positions.get(ad_idx, (None, None))
            if ad_col is not None and ad_row is not None:
                ax.plot(
                    [ad_col, hit_col],
                    [ad_row, hit_row],
                    color=AD_KILL_LINK,
                    linestyle=":",
                    linewidth=1.4,
                    alpha=0.85,
                    zorder=3,
                )
            ax.scatter(
                hit_col,
                hit_row,
                facecolors=AD_KILL_COLOR,
                edgecolors=AD_KILL_EDGE,
                marker="X",
                s=180,
                linewidths=1.3,
                label=label,
                zorder=10,
            )
            ax.text(
                hit_col + 0.1,
                hit_row + 0.1,
                f"AD{ad_idx}→D{idx}\nT={intercept_time:.1f}",
                color="black",
                fontsize=7,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )
        if drone["destroyed_by"] == "interceptor" and drone.get("interceptor_hit"):
            hit_row, hit_col = drone["interceptor_hit"]
            label = None
            if not interceptor_kill_label_added:
                label = "Interceptor kill"
                interceptor_kill_label_added = True
            ax.scatter(
                hit_col,
                hit_row,
                color=INTERCEPTOR_KILL_COLOR,
                marker="*",
                s=120,
                linewidths=1.5,
                edgecolors="black",
                label=label,
            )
            if drone.get("interceptor_time") is not None:
                ax.text(
                    hit_col + 0.1,
                    hit_row - 0.3,
                    f"t={drone['interceptor_time']:.1f}",
                    color="tab:cyan",
                    fontsize=7,
                )

        ax.text(
            0.02,
            0.05,
            f"AD kills: {ad_kills}\nInterceptor kills: {interceptor_kills}\nAD-target strikes: {ad_attrit}\nSurvivors: {survivors}",
            transform=ax.transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )

    ax.legend(loc="upper right")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


@dataclass
class DroneSeries:
    """Track a drone's path over time for animation."""
    entry: Tuple[float, float]
    destination: Tuple[float, float]
    tot: float
    start_time: float
    times: List[float]
    points: List[Tuple[float, float]]
    destroyed_time: Optional[float]
    color: str
    kill_point: Optional[Tuple[float, float]] = None
    kill_type: Optional[str] = None
    interceptor_hit: Optional[Tuple[float, float]] = None
    interceptor_time: Optional[float] = None


def _compute_drone_path(entry: Tuple[int, int], destination: Tuple[int, int], 
                        tot: float, kill_time: Optional[float] = None,
                        kill_point: Optional[Tuple[float, float]] = None) -> DroneSeries:
    """Compute drone path from entry to destination (or kill point)."""
    entry_row, entry_col = float(entry[0]), float(entry[1])
    dest_row, dest_col = float(destination[0]), float(destination[1])
    
    # If killed, use kill point as destination
    if kill_point is not None:
        dest_row, dest_col = kill_point[0], kill_point[1]
    
    # Compute total distance
    total_distance = math.dist((entry_row, entry_col), (dest_row, dest_col))
    
    # Generate path samples (interpolate along the line)
    num_samples = max(10, int(total_distance * 2))  # At least 10 samples
    times = []
    points = []
    
    for i in range(num_samples + 1):
        ratio = i / num_samples if num_samples > 0 else 0.0
        row = entry_row + (dest_row - entry_row) * ratio
        col = entry_col + (dest_col - entry_col) * ratio
        distance = ratio * total_distance
        time = tot + distance / DRONE_SPEED
        
        # Stop at kill time if specified
        if kill_time is not None and time >= kill_time:
            # Interpolate to exact kill point
            if kill_point is not None:
                points.append(kill_point)
                times.append(kill_time)
            break
        
        points.append((row, col))
        times.append(time)
    
    # If not killed, ensure we reach destination
    if kill_time is None or (times and times[-1] < tot + total_distance / DRONE_SPEED):
        points.append((dest_row, dest_col))
        times.append(tot + total_distance / DRONE_SPEED)
    
    return DroneSeries(
        entry=(entry_row, entry_col),
        destination=(dest_row, dest_col),
        tot=tot,
        start_time=tot,
        times=times,
        points=points,
        destroyed_time=kill_time,
        color=TOT_PALETTE.get(tot, "tab:red"),
        kill_point=kill_point,
        kill_type=None,  # Will be set separately
        interceptor_hit=None,
        interceptor_time=None,
    )


def _prepare_drone_series(snapshot: Dict[str, object]) -> Tuple[List[DroneSeries], Dict[int, int]]:
    """Prepare drone series for animation from snapshot.
    
    Returns:
        Tuple of (drone_series_list, drone_to_ad_map) where drone_to_ad_map maps
        drone index to AD index for AD kills
    """
    drones: Sequence[Dict[str, object]] = snapshot["drones"]
    series: List[DroneSeries] = []
    drone_to_ad: Dict[int, int] = {}  # Map drone idx to AD idx for AD kills
    
    for idx, drone in enumerate(drones):
        entry = (float(drone["entry"][0]), float(drone["entry"][1]))
        destination = tuple(drone["destination"])
        tot = float(drone.get("tot", 0.0))
        
        # Determine kill information
        destroyed_by = str(drone.get("destroyed_by") or "")
        kill_time = None
        kill_point = None
        kill_type = None
        interceptor_hit = None
        interceptor_time = None
        
        # Check for AD intercept
        intercepts = drone.get("intercepts", ())
        if intercepts and destroyed_by.startswith("ad:"):
            ad_idx, hit_point, intercept_time = intercepts[0]
            kill_time = float(intercept_time)
            kill_point = (float(hit_point[0]), float(hit_point[1]))
            kill_type = "ad"
            drone_to_ad[idx] = ad_idx
        
        # Check for interceptor kill
        elif destroyed_by.startswith("interceptor"):
            interceptor_hit = drone.get("interceptor_hit")
            interceptor_time = drone.get("interceptor_time")
            if interceptor_hit is not None and interceptor_time is not None:
                kill_time = float(interceptor_time)
                kill_point = (float(interceptor_hit[0]), float(interceptor_hit[1]))
                kill_type = "interceptor"
        
        # Compute path
        drone_series = _compute_drone_path(entry, destination, tot, kill_time, kill_point)
        drone_series.kill_type = kill_type
        drone_series.interceptor_hit = interceptor_hit
        drone_series.interceptor_time = interceptor_time
        series.append(drone_series)
    
    return series, drone_to_ad


def _position_at(drone: DroneSeries, t: float) -> Tuple[float, float]:
    """Get drone position at time t."""
    if t <= drone.start_time:
        return drone.entry
    if not drone.times:
        return drone.entry
    
    # Find the segment containing time t
    for i in range(len(drone.times) - 1):
        if drone.times[i] <= t <= drone.times[i + 1]:
            # Interpolate between points
            ratio = (t - drone.times[i]) / (drone.times[i + 1] - drone.times[i])
            ratio = min(max(ratio, 0.0), 1.0)
            p0 = drone.points[i]
            p1 = drone.points[i + 1]
            return (
                p0[0] + (p1[0] - p0[0]) * ratio,
                p0[1] + (p1[1] - p0[1]) * ratio,
            )
    
    # Beyond last point
    return drone.points[-1]


def _prefix_path(drone: DroneSeries, t: float) -> List[Tuple[float, float]]:
    """Get path prefix up to time t."""
    points = [drone.entry]
    for i, time in enumerate(drone.times):
        if time <= t:
            points.append(drone.points[i])
        else:
            # Interpolate to current position
            if i > 0:
                ratio = (t - drone.times[i - 1]) / (drone.times[i] - drone.times[i - 1])
                ratio = min(max(ratio, 0.0), 1.0)
                p0 = drone.points[i - 1]
                p1 = drone.points[i]
                points.append((
                    p0[0] + (p1[0] - p0[0]) * ratio,
                    p0[1] + (p1[1] - p0[1]) * ratio,
                ))
            break
    return points


def _max_time(series: Sequence[DroneSeries]) -> float:
    """Get maximum time across all drones."""
    max_t = 0.0
    for drone in series:
        if drone.times:
            max_t = max(max_t, drone.times[-1])
        else:
            max_t = max(max_t, drone.start_time)
    return max_t + 2.0  # Add buffer


def _target_events(snapshot: Dict[str, object]) -> List[Dict[str, object]]:
    """Extract target destruction events."""
    targets = snapshot.get("targets", ())
    destroyed_flags = snapshot.get("target_destroyed", ())
    drones = snapshot.get("drones", ())
    events = []
    
    for idx, target in enumerate(targets):
        destroyed_time = None
        if idx < len(destroyed_flags) and destroyed_flags[idx]:
            # Find earliest arrival time for this target
            arrival_times = []
            for drone in drones:
                if (drone.get("strike_success") and 
                    drone.get("target_idx") == idx):
                    entry = drone["entry"]
                    dest = drone["destination"]
                    tot = float(drone.get("tot", 0.0))
                    arrival_time = tot + math.dist(entry, dest) / DRONE_SPEED
                    arrival_times.append(arrival_time)
            if arrival_times:
                destroyed_time = min(arrival_times)
        
        events.append({
            "row": getattr(target, "row", 0.0),
            "col": getattr(target, "col", 0.0),
            "value": getattr(target, "value", 0.0),
            "destroyed_time": destroyed_time,
        })
    
    return events


def _status_counts(series: Sequence[DroneSeries], target_events: Sequence[Dict[str, object]], 
                   t: float) -> Dict[str, int]:
    """Count statuses at time t."""
    counts = {
        "targets": 0,
        "ad": 0,
        "interceptor": 0,
        "active": 0,
        "survivors": 0,
    }
    
    # Count destroyed targets
    counts["targets"] = sum(
        1 for event in target_events
        if event.get("destroyed_time") is not None
        and t >= float(event["destroyed_time"])
    )
    
    # Count drone outcomes
    for drone in series:
        resolved = drone.destroyed_time is not None and t >= drone.destroyed_time
        if resolved:
            if drone.kill_type == "ad":
                counts["ad"] += 1
            elif drone.kill_type == "interceptor":
                counts["interceptor"] += 1
        else:
            counts["active"] += 1
    
    counts["survivors"] = counts["active"]
    counts["destroyed"] = len(series) - counts["active"]
    
    return counts


def render_animation(state: SwarmDefenseState, output_path: Path, 
                     time_step: float = 0.25, fps: int = 12) -> None:
    """Render animated visualization of the game episode.
    
    Args:
        state: Final game state
        output_path: Path to save animation GIF
        time_step: Simulation timestep in seconds
        fps: Frames per second for animation
    """
    snapshot = state.snapshot()
    config = state.config
    grid_rows = config.grid_rows
    grid_cols = config.grid_cols
    bottom_half_start = config.bottom_half_start
    ad_position_candidates = config.get_ad_position_candidates()
    
    # Prepare data
    series, drone_to_ad = _prepare_drone_series(snapshot)
    target_events = _target_events(snapshot)
    ad_units = snapshot["ad_units"]
    drones = snapshot["drones"]
    
    print(f"\n📊 Animation data:")
    print(f"  - Drones in snapshot: {len(drones)}")
    print(f"  - Drone series prepared: {len(series)}")
    print(f"  - Targets: {len(target_events)}")
    print(f"  - AD units: {len(ad_units)}")
    
    # Debug: Print drone entry positions
    for idx, ser in enumerate(series):
        print(f"  - Drone {idx}: entry=({ser.entry[0]:.1f}, {ser.entry[1]:.1f}), "
              f"dest=({ser.destination[0]:.1f}, {ser.destination[1]:.1f}), "
              f"tot={ser.tot:.1f}, killed={ser.destroyed_time is not None}")
    
    max_t = _max_time(series)
    frames = int(math.ceil(max_t / time_step)) + 1
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    grid = np.zeros((grid_rows, grid_cols))
    ax.imshow(grid, cmap="Greys", alpha=0.05, extent=(-0.5, grid_cols - 0.5, grid_rows - 0.5, -0.5))
    ax.set_xticks(range(grid_cols))
    ax.set_yticks(range(grid_rows))
    ax.grid(True, which="both", color="lightgrey", linewidth=0.5, alpha=0.5)
    ax.set_xlim(-0.5, grid_cols - 0.5)
    ax.set_ylim(grid_rows - 0.5, -0.5)
    ax.set_title("Swarm Defense Episode Animation")
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")
    
    # Draw bottom-half area
    ax.add_patch(
        plt.Rectangle(
            (-0.5, bottom_half_start - 0.5),
            grid_cols,
            grid_rows - bottom_half_start,
            linewidth=1.0,
            edgecolor="tab:blue",
            facecolor="none",
            linestyle="--",
            label="Bottom-half AO",
        )
    )
    
    # Draw AD candidate cells
    for row, col in ad_position_candidates:
        ax.scatter(col, row, s=10, color="tab:blue", alpha=0.3)
    
    # Initialize target markers
    target_base_markers = []
    target_kill_markers = []
    for idx, target in enumerate(target_events):
        base = ax.scatter(target["col"], target["row"], s=220, color="tab:green", 
                         marker="o", zorder=3, label="Target" if idx == 0 else "")
        target_base_markers.append(base)
        kill = ax.scatter(
            target["col"], target["row"], s=260, marker=TARGET_KILL_MARKER,
            color=TARGET_KILL_COLOR, edgecolors=TARGET_KILL_EDGE, linewidths=1.6,
            alpha=0.0, zorder=6, label="Destroyed target" if idx == 0 else ""
        )
        target_kill_markers.append(kill)
        ax.text(target["col"] + 0.3, target["row"], f"T{idx}\nV={target['value']:.0f}", 
                fontsize=9)
    
    # Initialize AD unit markers
    ad_markers = []
    ad_positions: Dict[int, Tuple[float, float]] = {}
    for idx, unit in enumerate(ad_units):
        col, row = unit["position"][1], unit["position"][0]
        ad_positions[idx] = (col, row)
        alive = bool(unit["alive"])
        color = "tab:blue" if alive else "tab:gray"
        marker = "^" if alive else "v"
        ad_marker = ax.scatter(col, row, marker=marker, s=200, color=color, 
                              zorder=4, label="AD unit" if idx == 0 else "")
        ad_markers.append(ad_marker)
        ax.text(col - 0.6, row - 0.4, f"AD{idx}", color=color, fontsize=8)
    
    # Initialize drone markers and paths
    drone_markers = []
    path_lines = []
    kill_markers = []
    kill_pulses = []
    ad_kill_lines = []  # Lines from AD units to kill points
    
    for idx, ser in enumerate(series):
        # Drone marker
        marker = ax.scatter(ser.entry[1], ser.entry[0], color=ser.color, 
                           marker="s", s=80, zorder=5, label="Drone entry" if idx == 0 else "",
                           alpha=1.0)  # Explicitly set alpha to 1.0
        drone_markers.append(marker)
        print(f"  Created marker for drone {idx} at ({ser.entry[1]:.1f}, {ser.entry[0]:.1f})")
        
        # Path line
        line = ax.plot([], [], color=ser.color, linewidth=2, alpha=0.7, zorder=2)[0]
        path_lines.append(line)
        
        # Kill marker and pulse
        if ser.kill_point is not None:
            if ser.kill_type == "ad":
                kill_marker = ax.scatter(
                    ser.kill_point[1], ser.kill_point[0],
                    facecolors=AD_KILL_COLOR, edgecolors=AD_KILL_EDGE,
                    marker="X", s=150, linewidths=1.2, alpha=0.0, zorder=6,
                    label="AD kill" if idx == 0 else ""
                )
                # Add AD kill link line (will be shown when kill happens)
                ad_kill_line = ax.plot([], [], color=AD_KILL_LINK, linestyle=":",
                                       linewidth=1.4, alpha=0.0, zorder=3)[0]
                ad_kill_lines.append(ad_kill_line)
            elif ser.kill_type == "interceptor":
                kill_marker = ax.scatter(
                    ser.kill_point[1], ser.kill_point[0],
                    color=INTERCEPTOR_KILL_COLOR, marker="*", s=140,
                    linewidths=1.2, edgecolors="black", alpha=0.0, zorder=6,
                    label="Interceptor kill" if idx == 0 else ""
                )
                # Not an AD kill, so no AD kill line
                ad_kill_lines.append(None)
            else:
                kill_marker = ax.scatter(
                    ser.kill_point[1], ser.kill_point[0],
                    color="tab:red", marker="D", s=130,
                    linewidths=1.0, edgecolors="black", alpha=0.0, zorder=6
                )
                # Not an AD kill, so no AD kill line
                ad_kill_lines.append(None)
            kill_markers.append(kill_marker)
            
            # Pulse effect
            pulse = patches.Circle(
                (ser.kill_point[1], ser.kill_point[0]), radius=0.0,
                fill=False, linewidth=1.5, edgecolor=ser.color,
                alpha=0.0, zorder=5
            )
            ax.add_patch(pulse)
            kill_pulses.append(pulse)
        else:
            kill_markers.append(None)
            kill_pulses.append(None)
            ad_kill_lines.append(None)
    
    # Status text
    status_text = ax.text(
        0.01, 0.02, "", transform=ax.transAxes, fontsize=10,
        va="bottom", ha="left",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"}
    )
    
    # Animation update function
    def _update(frame_idx: int):
        current_time = frame_idx * time_step
        counts = _status_counts(series, target_events, current_time)
        
        # Update status text
        status_text.set_text(
            f"t={current_time:.1f}s\n"
            f"Targets destroyed: {counts['targets']}\n"
            f"AD kills: {counts['ad']}\n"
            f"Interceptor kills: {counts['interceptor']}\n"
            f"Active drones: {counts['active']}\n"
            f"Survivors: {counts['survivors']}"
        )
        
        # Update targets
        for idx, target in enumerate(target_events):
            destroyed_time = target.get("destroyed_time")
            killed = destroyed_time is not None and current_time >= float(destroyed_time)
            target_base_markers[idx].set_alpha(0.0 if killed else 1.0)
            target_kill_markers[idx].set_alpha(1.0 if killed else 0.0)
        
        # Update drones
        for idx, ser in enumerate(series):
            if idx >= len(drone_markers):
                print(f"  WARNING: Drone {idx} has no marker! (only {len(drone_markers)} markers)")
                continue
            pos = _position_at(ser, current_time)
            drone_markers[idx].set_offsets([[pos[1], pos[0]]])
            drone_markers[idx].set_alpha(1.0)  # Ensure visible
            
            # Update path
            prefix = _prefix_path(ser, current_time)
            xs = [p[1] for p in prefix]
            ys = [p[0] for p in prefix]
            path_lines[idx].set_data(xs, ys)
            path_lines[idx].set_alpha(0.7)  # Ensure visible
            
            # Update kill markers
            if ser.destroyed_time is not None and current_time >= ser.destroyed_time:
                drone_markers[idx].set_alpha(0.4)
                drone_markers[idx].set_edgecolor("black")
                drone_markers[idx].set_facecolor("none")
                
                if kill_markers[idx] is not None:
                    kill_markers[idx].set_alpha(1.0)
                
                # Update AD kill link
                if (idx in drone_to_ad and idx < len(ad_kill_lines) and 
                    ad_kill_lines[idx] is not None):
                    ad_idx = drone_to_ad[idx]
                    if ad_idx in ad_positions and ser.kill_point is not None:
                        ad_col, ad_row = ad_positions[ad_idx]
                        kill_col, kill_row = ser.kill_point[1], ser.kill_point[0]
                        ad_kill_lines[idx].set_data([ad_col, kill_col], [ad_row, kill_row])
                        ad_kill_lines[idx].set_alpha(0.85)
                
                # Update pulse
                if kill_pulses[idx] is not None:
                    pulse_age = current_time - ser.destroyed_time
                    if pulse_age <= 1.5:
                        pulse = kill_pulses[idx]
                        pulse.set_alpha(max(0.0, 1.0 - pulse_age / 1.5))
                        pulse.set_radius(0.2 + pulse_age * 0.8)
                    else:
                        kill_pulses[idx].set_alpha(0.0)
                        kill_pulses[idx].set_radius(0.0)
            else:
                if kill_markers[idx] is not None:
                    kill_markers[idx].set_alpha(0.0)
                if kill_pulses[idx] is not None:
                    kill_pulses[idx].set_alpha(0.0)
                    kill_pulses[idx].set_radius(0.0)
                if idx < len(ad_kill_lines) and ad_kill_lines[idx] is not None:
                    ad_kill_lines[idx].set_alpha(0.0)
        
        # Collect all artists for blitting
        artists = (
            drone_markers + path_lines + target_base_markers + target_kill_markers +
            ad_markers + [status_text] +
            [m for m in kill_markers if m is not None] +
            [p for p in kill_pulses if p is not None] +
            [l for l in ad_kill_lines if l is not None]
        )
        return artists
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, _update, frames=frames, interval=1000 / fps, blit=False, repeat=True
    )
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    anim.save(output_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


def load_config_file(config_path: Path) -> Dict:
    """Load game parameters from YAML config file."""
    if yaml is None:
        raise RuntimeError("PyYAML is required to load configuration files (pip install pyyaml).")
    
    expanded = config_path.expanduser()
    if not expanded.exists():
        raise FileNotFoundError(f"Config file not found: {expanded}")
    
    with expanded.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    
    if data is None:
        return {}
    
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file {expanded} must contain a top-level mapping.")
    
    # Extract game parameters from 'game' section
    game_params_section = (
        data.get("game")
        or data.get("game_parameters")
        or data.get("game_params")
    )
    
    if game_params_section is not None and not isinstance(game_params_section, dict):
        raise ValueError("`game` configuration must be provided as a mapping.")
    
    return dict(game_params_section or {})


def load_game_params_from_metadata(path: Path) -> Optional[Dict]:
    """Load game parameters from metadata.pt.
    
    Args:
        path: Path to metadata.pt file, policy.pt file, or checkpoint directory
    
    Returns:
        Dictionary of game parameters, or None if metadata not found
    """
    # Determine the metadata.pt path
    if path.name == "metadata.pt" and path.is_file():
        # Direct path to metadata.pt
        metadata_path = path
    elif path.name == "policy.pt" and path.is_file():
        # Path to policy.pt, look for metadata.pt in same directory
        metadata_path = path.parent / "metadata.pt"
    elif path.is_dir():
        # Path to checkpoint directory
        metadata_path = path / "metadata.pt"
    else:
        # Try as file and get parent directory
        if path.is_file():
            metadata_path = path.parent / "metadata.pt"
        else:
            # Assume it's a directory even if is_dir() is False (might be a symlink)
            metadata_path = path / "metadata.pt"
    
    if not metadata_path.exists():
        return None
    
    try:
        metadata = torch.load(str(metadata_path), map_location="cpu")
        game_config = metadata.get('game_config', {})
        game_params = game_config.get('parameters', {})
        
        if game_params:
            return dict(game_params)
    except Exception as e:
        print(f"Warning: Could not load metadata from {metadata_path}: {e}")
        return None
    
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run and render a Swarm Defense episode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simplest: Use checkpoint directory (auto-loads metadata.pt and policy.pt)
  python demo_visualizer.py --checkpoint ../ESCHER-Torch/results/swarm_defense/2025_11_16_04_57_09/
  
  # Use checkpoint with animation
  python demo_visualizer.py --checkpoint ../ESCHER-Torch/results/swarm_defense/2025_11_16_04_57_09/ --animate
  
  # Override game parameters from checkpoint
  python demo_visualizer.py --checkpoint ../ESCHER-Torch/results/.../ --num_targets 3
  
  # Use config file
  python demo_visualizer.py --config ../ESCHER-Torch/configs/swarm_example.yaml
  
  # Use command line arguments
  python demo_visualizer.py --grid_rows 8 --grid_cols 6 --num_targets 2
  
  # Combine config file with command line overrides
  python demo_visualizer.py --config config.yaml --num_targets 3
  
  # Use trained policy - game config automatically loaded from metadata.pt
  python demo_visualizer.py --policy ../ESCHER-Torch/results/.../policy.pt
  
  # Use separate policies for each player (config from first policy's metadata)
  python demo_visualizer.py --policy-p0 attacker.pt --policy-p1 defender.pt
  
  # Use policy with sampling (instead of argmax)
  python demo_visualizer.py --checkpoint ../ESCHER-Torch/results/.../ --sampling
  
  # Customize animation settings
  python demo_visualizer.py --checkpoint ../ESCHER-Torch/results/.../ --animate --time-step 0.1 --fps 24
        """
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducible runs")
    parser.add_argument("--output", type=str, default=None, help="Output path for visualization (default: Visualizer/swarm_defense_demo.png)")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file (game parameters will be read from 'game:' section). If omitted, config will be loaded from metadata.pt (from --checkpoint or --policy)")
    parser.add_argument("--grid_rows", type=int, default=None, help="Grid rows (overrides config file)")
    parser.add_argument("--grid_cols", type=int, default=None, help="Grid columns (overrides config file)")
    parser.add_argument("--num_targets", type=int, default=None, help="Number of targets (overrides config file)")
    parser.add_argument("--num_ad_units", type=int, default=None, help="Number of AD units (overrides config file)")
    parser.add_argument("--num_attacking_drones", type=int, default=None, help="Number of attacking drones (overrides config file)")
    parser.add_argument("--num_interceptors", type=int, default=None, help="Number of interceptors (overrides config file)")
    parser.add_argument("--ad_kill_probability", type=float, default=None, help="AD kill probability (overrides config file)")
    parser.add_argument("--interceptor_reward", type=float, default=None, help="Interceptor reward (overrides config file)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint directory (will auto-load metadata.pt and policy.pt from this directory)")
    parser.add_argument("--policy", type=str, default=None, help="Path to policy.pt file (for both players, or use --policy-p0/--policy-p1 for separate policies). If --checkpoint is provided, this is ignored.")
    parser.add_argument("--policy-p0", type=str, default=None, help="Path to policy.pt file for Player 0 (Attacker). If --checkpoint is provided, this is ignored.")
    parser.add_argument("--policy-p1", type=str, default=None, help="Path to policy.pt file for Player 1 (Defender). If --checkpoint is provided, this is ignored.")
    parser.add_argument("--sampling", action="store_true", help="Sample from policy distribution instead of taking argmax")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device for policy inference (cpu or cuda)")
    parser.add_argument("--animate", action="store_true", help="Create animated GIF instead of static snapshot")
    parser.add_argument("--animation-output", type=str, default=None, help="Output path for animation GIF (default: Visualizer/swarm_defense_animation.gif)")
    parser.add_argument("--time-step", type=float, default=0.25, help="Simulation timestep in seconds for animation")
    parser.add_argument("--fps", type=int, default=12, help="Frames per second for animation")
    args = parser.parse_args()

    # Handle checkpoint directory (highest priority for auto-loading)
    checkpoint_dir = None
    if args.checkpoint:
        checkpoint_dir = Path(args.checkpoint)
        if not checkpoint_dir.exists():
            print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
            sys.exit(1)
        if not checkpoint_dir.is_dir():
            print(f"Error: Checkpoint path is not a directory: {checkpoint_dir}")
            sys.exit(1)
        
        # Check for required files
        metadata_path = checkpoint_dir / "metadata.pt"
        policy_path = checkpoint_dir / "policy.pt"
        
        if not metadata_path.exists():
            print(f"Error: metadata.pt not found in checkpoint directory: {checkpoint_dir}")
            sys.exit(1)
        if not policy_path.exists():
            print(f"Error: policy.pt not found in checkpoint directory: {checkpoint_dir}")
            sys.exit(1)
        
        print(f"📁 Using checkpoint directory: {checkpoint_dir}")
    
    # Load game parameters - priority: config file > metadata from checkpoint > metadata from policy > defaults
    game_params = {}
    
    # First, try to load from config file
    if args.config:
        config_path = Path(args.config)
        game_params = load_config_file(config_path)
        print(f"📄 Loaded game parameters from config file: {config_path}")
        for key, value in game_params.items():
            print(f"   - {key}: {value}")
    
    # If no config file, try to load from checkpoint directory
    elif checkpoint_dir:
        metadata_path = checkpoint_dir / "metadata.pt"
        metadata_params = load_game_params_from_metadata(metadata_path)
        if metadata_params:
            game_params = metadata_params
            print(f"📄 Loaded game parameters from metadata.pt (from checkpoint: {checkpoint_dir})")
            for key, value in game_params.items():
                print(f"   - {key}: {value}")
        else:
            print(f"Warning: Could not load game parameters from {metadata_path}")
    
    # If no config file and no checkpoint, try to load from metadata.pt (from policy paths)
    elif args.policy or args.policy_p0 or args.policy_p1:
        # Try to get metadata from the policy path(s)
        policy_paths = []
        if args.policy:
            policy_paths.append(Path(args.policy))
        if args.policy_p0:
            policy_paths.append(Path(args.policy_p0))
        if args.policy_p1:
            policy_paths.append(Path(args.policy_p1))
        
        # Try each policy path to find metadata
        for policy_path in policy_paths:
            metadata_params = load_game_params_from_metadata(policy_path)
            if metadata_params:
                game_params = metadata_params
                print(f"📄 Loaded game parameters from metadata.pt (from policy: {policy_path})")
                for key, value in game_params.items():
                    print(f"   - {key}: {value}")
                break
    
    # Override with command line arguments (command line takes precedence)
    if args.grid_rows is not None:
        game_params["grid_rows"] = args.grid_rows
    if args.grid_cols is not None:
        game_params["grid_cols"] = args.grid_cols
    if args.num_targets is not None:
        game_params["num_targets"] = args.num_targets
    if args.num_ad_units is not None:
        game_params["num_ad_units"] = args.num_ad_units
    if args.num_attacking_drones is not None:
        game_params["num_attacking_drones"] = args.num_attacking_drones
    if args.num_interceptors is not None:
        game_params["num_interceptors"] = args.num_interceptors
    if args.ad_kill_probability is not None:
        game_params["ad_kill_probability"] = args.ad_kill_probability
    if args.interceptor_reward is not None:
        game_params["interceptor_reward"] = args.interceptor_reward
    
    # Use provided output path or default
    output_path = Path(args.output) if args.output else OUTPUT_PATH
    
    # Handle policy arguments - checkpoint takes precedence
    policy_path_p0 = None
    policy_path_p1 = None
    policy_layers = None  # Will try to load from metadata if available
    
    if checkpoint_dir:
        # Try to load policy architecture from metadata
        metadata_path = checkpoint_dir / "metadata.pt"
        if metadata_path.exists():
            try:
                metadata = torch.load(str(metadata_path), map_location="cpu")
                # Check if policy_layers is stored in metadata (future-proofing)
                if "policy_layers" in metadata:
                    policy_layers = tuple(metadata["policy_layers"])
                    print(f"📐 Policy architecture from metadata: {policy_layers}")
            except Exception as e:
                print(f"Warning: Could not load policy architecture from metadata: {e}")
        
        # Auto-load policy from checkpoint directory
        policy_path = checkpoint_dir / "policy.pt"
        policy_path_p0 = policy_path
        policy_path_p1 = policy_path
        print(f"📦 Auto-loading policy from checkpoint: {policy_path}")
    elif args.policy:
        # Use same policy for both players
        policy_path_p0 = Path(args.policy)
        policy_path_p1 = Path(args.policy)
        if not policy_path_p0.exists():
            print(f"Error: Policy file not found: {policy_path_p0}")
            sys.exit(1)
    else:
        # Use separate policies
        if args.policy_p0:
            policy_path_p0 = Path(args.policy_p0)
            if not policy_path_p0.exists():
                print(f"Error: Player 0 policy file not found: {policy_path_p0}")
                sys.exit(1)
        if args.policy_p1:
            policy_path_p1 = Path(args.policy_p1)
            if not policy_path_p1.exists():
                print(f"Error: Player 1 policy file not found: {policy_path_p1}")
                sys.exit(1)
    
    # Print policy info
    if policy_path_p0 or policy_path_p1:
        print(f"\n🤖 Policy Configuration:")
        print(f"   Player 0 (Attacker): {'Trained policy' if policy_path_p0 else 'Default heuristic'}")
        print(f"   Player 1 (Defender): {'Trained policy' if policy_path_p1 else 'Default heuristic'}")
        print(f"   Sampling: {'Yes' if args.sampling else 'No (argmax)'}")
        print(f"   Device: {args.device}")
        print()

    state, seed = play_episode(
        args.seed,
        game_params if game_params else None,
        policy_path_p0,
        policy_path_p1,
        args.sampling,
        args.device,
        policy_layers,
    )
    
    # Render visualization
    if args.animate:
        animation_output = Path(args.animation_output) if args.animation_output else ANIMATION_OUTPUT_PATH
        print(f"\n🎬 Creating animation...")
        print(f"   Time step: {args.time_step}s")
        print(f"   FPS: {args.fps}")
        render_animation(state, animation_output, args.time_step, args.fps)
        print(f"✓ Animation saved to: {animation_output}")
    else:
        render_snapshot(state, output_path)
        print(f"✓ Snapshot saved to: {output_path}")
    
    returns = state.returns()
    ad_kills, interceptor_kills, survivors, ad_attrit = _count_outcomes(state.snapshot()["drones"])
    print("\nEpisode complete.")
    print(f"Seed used: {seed}")
    print(f"Attacker damage: {returns[0]:.1f}")
    print(f"Defender reward: {returns[1]:.1f}")
    print(
        "Breakdown -> AD kills: {ad} (intercepts), AD-target strikes: {attrit}, "
        "Interceptor kills: {inter}, Survivors: {surv}".format(
            ad=ad_kills, attrit=ad_attrit, inter=interceptor_kills, surv=survivors
        )
    )


if __name__ == "__main__":
    main()
