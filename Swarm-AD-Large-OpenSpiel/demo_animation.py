"""Time-stepped animation for the large Swarm Defense scenario with learned policy."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib import animation, patches
import numpy as np
import torch
import pyspiel

# Add paths for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "Swarm-AD-OpenSpiel"))
sys.path.insert(0, str(PROJECT_ROOT / "ESCHER-Torch"))

import swarm_defense_game as small_game
from ESCHER_Torch.eschersolver import PolicyNetwork
from swarm_defense_large_game import (
    SwarmDefenseLargeGame,
    SwarmDefenseLargeState,
    Phase,
    encode_drone_action,
    AD_COVERAGE_RADIUS,
    ARENA_HEIGHT,
    ARENA_WIDTH,
    LARGE_TOT_CHOICES,
    NUM_ATTACKING_DRONES,
    NUM_INTERCEPTORS,
    TARGET_POSITION_ACTION_BASE,
    TARGET_VALUE_ACTION_BASE,
    AD_ACTION_BASE,
    DRONE_ACTION_BASE,
    INTERCEPT_ACTION_BASE,
    TARGET_CANDIDATE_CELLS,
    ENTRY_POINTS,
)
from large_game_constants import ENTRY_LANES

OUTPUT_DIR = PROJECT_ROOT / "Visualizer"
OUTPUT_PATH = OUTPUT_DIR / "swarm_defense_large_animation.gif"
SNAPSHOT_OUTPUT_PATH = OUTPUT_DIR / "swarm_large_snapshot.json"
DRONE_SPEED = 1.0
AD_KILL_COLOR = "#FF3B30"
AD_KILL_EDGE = "black"
INTERCEPTOR_KILL_COLOR = "#0bc5ea"
AD_TARGET_KILL_COLOR = "#2E8B57"
AD_ROTATION_ARROW_COLOR = "#1f4c94"
INTERCEPTOR_ORIGIN = (30.0, 0.0)
INTERCEPTOR_VISUAL_DURATION = 6.0
TARGET_KILL_COLOR = "#F1C40F"
TARGET_KILL_EDGE = "#7D6608"
TARGET_KILL_MARKER = "P"


def _normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _interpolate_angle(start: float, end: float, ratio: float) -> float:
    delta = _normalize_angle(end - start)
    return start + delta * ratio


@dataclass
class DroneSeries:
    entry: Tuple[float, float]
    tot_offset: float
    start_time: float
    times: List[float]
    points: List[Tuple[float, float]]
    destroyed_time: Optional[float]
    color: str
    kill_point: Optional[Tuple[float, float]] = None
    kill_type: Optional[str] = None


def load_policy_network(checkpoint_path: Path) -> Tuple[torch.nn.Module, Dict]:
    """Load trained policy network and metadata."""
    policy_path = checkpoint_path / "policy.pt"
    
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy checkpoint not found at {policy_path}")
    
    # Use the small game config directly
    game = small_game.SwarmDefenseGame()
    initial_state = game.new_initial_state()
    state_size = len(initial_state.information_state_tensor(0))
    num_actions = game.num_distinct_actions()
    
    # Load metadata if available (optional)
    metadata = {}
    metadata_path = checkpoint_path / "metadata.pt"
    if metadata_path.exists():
        metadata = torch.load(metadata_path, map_location='cpu')
    
    # Build game config from small_game module
    game_config = {
        'GRID_SIZE': small_game.GRID_SIZE,
        'NUM_TARGETS': small_game.NUM_TARGETS,
        'NUM_AD_UNITS': small_game.NUM_AD_UNITS,
        'NUM_ATTACKING_DRONES': small_game.NUM_ATTACKING_DRONES,
        'NUM_INTERCEPTORS': small_game.NUM_INTERCEPTORS,
        'state_tensor_size': state_size,
        'num_distinct_actions': num_actions,
    }
    metadata['game_config'] = game_config
    
    # Create policy network matching the trained architecture
    hidden_layers = (256, 128)
    policy_net = PolicyNetwork(state_size, hidden_layers, num_actions)
    policy_net.load_state_dict(torch.load(policy_path, map_location='cpu'))
    policy_net.eval()
    
    return policy_net, metadata


def get_policy_action(policy_net: torch.nn.Module, state: pyspiel.State, 
                     legal_actions: List[int], num_actions: int) -> int:
    """Get action from policy network for the given state."""
    if not legal_actions:
        raise ValueError("No legal actions available")
    
    # Get state tensor
    state_tensor = torch.FloatTensor(state.information_state_tensor(state.current_player()))
    
    # Create mask for legal actions (1.0 for legal, 0.0 for illegal)
    mask = torch.zeros(num_actions, dtype=torch.float32)
    mask[legal_actions] = 1.0
    
    with torch.no_grad():
        probs = policy_net(state_tensor.unsqueeze(0), mask.unsqueeze(0)).squeeze(0)
    
    # Sample from policy
    action = torch.multinomial(probs, 1).item()
    
    return action


def run_small_game_with_policy(policy_net: torch.nn.Module, seed: int) -> Dict[str, object]:
    """Run the small grid game using the learned policy."""
    rng = random.Random(seed)
    game = small_game.SwarmDefenseGame()
    state = game.new_initial_state()
    num_actions = game.num_distinct_actions()
    
    while not state.is_terminal():
        player = state.current_player()
        
        if player == pyspiel.PlayerId.CHANCE:
            # Sample chance node
            outcomes = state.chance_outcomes()
            pick = rng.random()
            cumulative = 0.0
            for action, prob in outcomes:
                cumulative += prob
                if pick <= cumulative:
                    state.apply_action(action)
                    break
        else:
            # Use policy for player decisions
            legal_actions = state.legal_actions()
            action = get_policy_action(policy_net, state, legal_actions, num_actions)
            state.apply_action(action)
    
    return state.snapshot()


def grid_to_continuous(row: int, col: int) -> Tuple[float, float]:
    """Map discrete grid position to continuous arena position (cell center)."""
    grid_size = small_game.GRID_SIZE
    cell_height = ARENA_HEIGHT / grid_size
    cell_width = ARENA_WIDTH / grid_size
    
    # Place at center of grid cell
    continuous_row = (row + 0.5) * cell_height
    continuous_col = (col + 0.5) * cell_width
    
    return continuous_row, continuous_col


def find_closest_continuous_position(target_row: float, target_col: float) -> Tuple[float, float]:
    """Find the closest valid position in TARGET_CANDIDATE_CELLS."""
    min_dist = float('inf')
    closest_pos = TARGET_CANDIDATE_CELLS[0]
    
    for pos in TARGET_CANDIDATE_CELLS:
        dist = math.sqrt((pos[0] - target_row)**2 + (pos[1] - target_col)**2)
        if dist < min_dist:
            min_dist = dist
            closest_pos = pos
    
    return closest_pos


def map_small_to_large_game(small_snapshot: Dict[str, object], seed: int) -> SwarmDefenseLargeState:
    """Map small game snapshot to large game execution with 1:1 agent mapping."""
    rng = random.Random(seed)
    large_game = SwarmDefenseLargeGame()
    large_state = large_game.new_initial_state()
    
    # Extract small game data
    small_targets = small_snapshot["targets"]
    small_ad_units = small_snapshot["ad_units"]
    small_drones = small_snapshot["drones"]
    
    # Phase 1: Target positions (chance)
    target_positions_mapped = []
    for target in small_targets:
        # Map grid position to continuous
        continuous_pos = grid_to_continuous(int(target.row), int(target.col))
        # Find closest valid position
        closest_pos = find_closest_continuous_position(continuous_pos[0], continuous_pos[1])
        target_positions_mapped.append(closest_pos)
    
    # Apply target position actions
    for pos in target_positions_mapped:
        # Find action index for this position
        action_idx = TARGET_CANDIDATE_CELLS.index(pos)
        action = TARGET_POSITION_ACTION_BASE + action_idx
        if action in large_state.legal_actions():
            large_state.apply_action(action)
    
    # Phase 2: Target values (chance)
    if large_state.phase() == Phase.TARGET_VALUES:
        legal = large_state.legal_actions()
        action = rng.choice(legal)
        large_state.apply_action(action)
    
    # Phase 3: AD placement (defender)
    ad_positions_mapped = []
    for ad_unit in small_ad_units:
        # Map grid position to continuous
        continuous_pos = grid_to_continuous(int(ad_unit["position"][0]), 
                                           int(ad_unit["position"][1]))
        ad_positions_mapped.append(continuous_pos)
    
    # Apply AD placement actions
    for pos in ad_positions_mapped:
        if large_state.phase() != Phase.AD_PLACEMENT:
            break
        # Find closest valid AD position
        legal = large_state.legal_actions()
        best_action = None
        best_dist = float('inf')
        
        for action in legal:
            from swarm_defense_large_game import decode_ad_position_action
            candidate_pos = decode_ad_position_action(action)
            dist = math.sqrt((candidate_pos[0] - pos[0])**2 + (candidate_pos[1] - pos[1])**2)
            if dist < best_dist:
                best_dist = dist
                best_action = action
        
        if best_action is not None:
            large_state.apply_action(best_action)
    
    # Phase 4: Drone assignment (attacker) - 1:1 mapping
    for drone in small_drones:
        if large_state.phase() != Phase.SWARM_ASSIGNMENT:
            break
            
        entry_row, entry_col = drone["entry"]
        # Map small game column to large game entry lane
        entry_lane_idx = int((entry_col / small_game.GRID_SIZE) * ENTRY_LANES)
        entry_lane_idx = max(0, min(entry_lane_idx, ENTRY_LANES - 1))
        
        # Use default blueprint (direct path)
        blueprint_idx = 0
        
        action = encode_drone_action(
            entry_lane_idx,
            drone["target_idx"],
            drone["tot_idx"],
            blueprint_idx
        )
        
        if action in large_state.legal_actions():
            large_state.apply_action(action)
        else:
            # If exact action not legal, try to find similar one
            legal = large_state.legal_actions()
            if legal:
                large_state.apply_action(rng.choice(legal))
    
    # Phase 5: Interceptor assignment (defender) - 1:1 mapping
    interceptor_count = 0
    while large_state.phase() == Phase.INTERCEPT_ASSIGNMENT and interceptor_count < NUM_INTERCEPTORS:
        legal = large_state.legal_actions()
        if not legal:
            break
        
        # Randomly choose to engage or pass
        action = rng.choice(legal)
        large_state.apply_action(action)
        interceptor_count += 1
    
    # Phase 6+: Resolution phases (chance)
    while not large_state.is_terminal():
        if large_state.current_player() == pyspiel.PlayerId.CHANCE:
            outcomes = large_state.chance_outcomes()
            pick = rng.random()
            cumulative = 0.0
            for action, prob in outcomes:
                cumulative += prob
                if pick <= cumulative:
                    large_state.apply_action(action)
                    break
        else:
            # Shouldn't happen in normal flow
            legal = large_state.legal_actions()
            if legal:
                large_state.apply_action(rng.choice(legal))
            else:
                break
    
    return large_state


def _write_snapshot(snapshot: Dict[str, object], output_path: Path) -> Path:
    """Persist the large-game snapshot for downstream consumers (e.g., WinTAK)."""

    def _normalize_targets(raw_targets: Sequence[object]) -> List[Dict[str, float]]:
        normalized: List[Dict[str, float]] = []
        for target in raw_targets:
            if hasattr(target, "row"):
                normalized.append(
                    {
                        "row": float(getattr(target, "row")),
                        "col": float(getattr(target, "col")),
                        "value": float(getattr(target, "value")),
                    }
                )
            elif isinstance(target, dict):
                normalized.append(
                    {
                        "row": float(target.get("row", 0.0)),
                        "col": float(target.get("col", 0.0)),
                        "value": float(target.get("value", 0.0)),
                    }
                )
            elif isinstance(target, (list, tuple)) and len(target) >= 3:
                row, col, value = target[:3]
                normalized.append(
                    {
                        "row": float(row),
                        "col": float(col),
                        "value": float(value),
                    }
                )
        return normalized

    safe_snapshot = dict(snapshot)
    safe_snapshot["targets"] = _normalize_targets(snapshot.get("targets", ()))
    safe_snapshot["target_destroyed"] = list(snapshot.get("target_destroyed", ()))
    safe_snapshot["ad_units"] = [dict(unit) for unit in snapshot.get("ad_units", ())]
    safe_snapshot["drones"] = [dict(drone) for drone in snapshot.get("drones", ())]
    safe_snapshot["returns"] = list(snapshot.get("returns", ()))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(safe_snapshot, handle, indent=2, sort_keys=True)
    return output_path


def _tot_color(tot: float) -> str:
    palette = {
        LARGE_TOT_CHOICES[0]: "#ff4d4f",
        LARGE_TOT_CHOICES[1]: "#ffa940",
        LARGE_TOT_CHOICES[2]: "#ffec3d",
        LARGE_TOT_CHOICES[3]: "#9254de",
    }
    return palette.get(tot, "tab:red")


def _kill_event(drone: Dict[str, object], ad_positions: Dict[int, Tuple[float, float]]) -> Optional[Dict[str, object]]:
    destroyed = str(drone.get("destroyed_by") or "")
    hold = float(drone.get("hold_time") or 0.0)
    intercepts: Sequence[Tuple[int, Tuple[float, float], float]] = drone.get("intercepts", ())
    if destroyed.startswith("ad:") and intercepts:
        ad_idx, hit_point, intercept_time = intercepts[0]
        kill_time = float(intercept_time)
        kill_distance = max(0.0, (kill_time - hold) * DRONE_SPEED)
        return {
            "type": "ad",
            "point": tuple(hit_point),
            "time": kill_time,
            "distance": kill_distance,
        }
    if destroyed.startswith("interceptor"):
        hit = drone.get("interceptor_hit")
        kill_time = drone.get("interceptor_time")
        if hit is not None and kill_time is not None:
            kill_time = float(kill_time)
            kill_distance = max(0.0, (kill_time - hold) * DRONE_SPEED)
            return {
                "type": "interceptor",
                "point": tuple(hit),
                "time": kill_time,
                "distance": kill_distance,
            }
    if destroyed.startswith("ad_target:"):
        try:
            ad_idx = int(destroyed.split(":", maxsplit=1)[1])
        except (IndexError, ValueError):
            ad_idx = None
        ad_point = ad_positions.get(ad_idx)
        destination = drone.get("destination")
        entry = drone.get("entry")
        kill_point = tuple(ad_point or destination or entry)
        total_distance = float(drone.get("total_distance") or 0.0)
        kill_time = hold + total_distance / DRONE_SPEED
        return {
            "type": "ad_target",
            "point": kill_point,
            "time": kill_time,
            "distance": total_distance,
        }
    return None


def _prepare_drone_series(snapshot: Dict[str, object]) -> List[DroneSeries]:
    drones: Sequence[Dict[str, object]] = snapshot["drones"]  # type: ignore[assignment]
    ad_units: Sequence[Dict[str, object]] = snapshot["ad_units"]  # type: ignore[assignment]
    ad_positions: Dict[int, Tuple[float, float]] = {idx: tuple(unit["position"]) for idx, unit in enumerate(ad_units)}
    series: List[DroneSeries] = []
    for drone in drones:
        entry = tuple(drone["entry"])  # type: ignore[arg-type]
        tot = float(drone.get("tot") or 0.0)
        hold = float(drone.get("hold_time") or 0.0)
        samples: Sequence[Tuple[float, float, float]] = drone.get("path_samples", ())
        kill = _kill_event(drone, ad_positions)
        kill_distance = kill.get("distance") if isinstance(kill, dict) else None
        times: List[float] = []
        points: List[Tuple[float, float]] = []
        prev_sample: Optional[Tuple[float, float, float]] = None
        for row, col, dist in samples:
            times.append(hold + dist / DRONE_SPEED)
            points.append((row, col))
            if kill_distance is not None and dist >= kill_distance:
                if dist > kill_distance and prev_sample is not None and dist != prev_sample[2]:
                    ratio = (kill_distance - prev_sample[2]) / (dist - prev_sample[2])
                    ratio = min(max(ratio, 0.0), 1.0)
                    interp = (
                        prev_sample[0] + (row - prev_sample[0]) * ratio,
                        prev_sample[1] + (col - prev_sample[1]) * ratio,
                    )
                    points[-1] = interp
                    times[-1] = tot + kill_distance / DRONE_SPEED
                else:
                    times[-1] = hold + kill_distance / DRONE_SPEED
                break
            prev_sample = (row, col, dist)
        if not times:
            times = [hold]
            points = [entry]
        destroyed_time: Optional[float] = kill.get("time") if isinstance(kill, dict) else None
        kill_point = kill.get("point") if isinstance(kill, dict) else None
        kill_type = kill.get("type") if isinstance(kill, dict) else None
        color = _tot_color(tot)
        series.append(DroneSeries(entry, tot, hold, times, points, destroyed_time, color, kill_point, kill_type))
    return series


def _target_events(snapshot: Dict[str, object]) -> List[Dict[str, object]]:
    targets: Sequence[object] = snapshot.get("targets", ())  # type: ignore[assignment]
    destroyed_flags: Sequence[bool] = snapshot.get("target_destroyed", ())  # type: ignore[assignment]
    drones: Sequence[Dict[str, object]] = snapshot.get("drones", ())  # type: ignore[assignment]
    events: List[Dict[str, object]] = []
    for idx, target in enumerate(targets):
        destroyed_time: Optional[float] = None
        if idx < len(destroyed_flags) and destroyed_flags[idx]:
            arrival_times = [
                float(drone.get("arrival_time") or 0.0)
                for drone in drones
                if drone.get("strike_success") and drone.get("target_idx") == idx
            ]
            if arrival_times:
                destroyed_time = min(arrival_times)
        events.append({
            "row": getattr(target, "row", 0.0),
            "col": getattr(target, "col", 0.0),
            "value": getattr(target, "value", 0.0),
            "destroyed_time": destroyed_time,
        })
    return events


def _prefix_path(drone: DroneSeries, t: float) -> List[Tuple[float, float]]:
    points = [drone.entry]
    prev_time = drone.start_time
    prev_point = drone.entry
    for time, point in zip(drone.times, drone.points):
        if t >= time:
            points.append(point)
            prev_time = time
            prev_point = point
            continue
        if time == prev_time:
            break
        ratio = (t - prev_time) / (time - prev_time)
        ratio = min(max(ratio, 0.0), 1.0)
        interp = (
            prev_point[0] + (point[0] - prev_point[0]) * ratio,
            prev_point[1] + (point[1] - prev_point[1]) * ratio,
        )
        points.append(interp)
        break
    return points


def _position_at(drone: DroneSeries, t: float) -> Tuple[float, float]:
    if t <= drone.start_time:
        return drone.entry
    prev_time = drone.start_time
    prev_point = drone.entry
    for time, point in zip(drone.times, drone.points):
        if t <= time:
            if time == prev_time:
                return point
            ratio = (t - prev_time) / (time - prev_time)
            ratio = min(max(ratio, 0.0), 1.0)
            return (
                prev_point[0] + (point[0] - prev_point[0]) * ratio,
                prev_point[1] + (point[1] - prev_point[1]) * ratio,
            )
        prev_time = time
        prev_point = point
    return drone.points[-1]


def _max_time(series: Sequence[DroneSeries]) -> float:
    max_t = 0.0
    for drone in series:
        max_t = max(max_t, drone.times[-1] if drone.times else drone.tot)
    return max_t + 2.0


def _status_counts(
    series: Sequence[DroneSeries],
    target_events: Sequence[Dict[str, object]],
    t: float,
) -> Dict[str, int]:
    counts = {
        "targets": 0,
        "ad": 0,
        "interceptor": 0,
        "active": 0,
    }
    counts["targets"] = sum(
        1
        for event in target_events
        if event.get("destroyed_time") is not None
        and t >= float(event["destroyed_time"])
    )
    for drone in series:
        resolved = drone.destroyed_time is not None and t >= drone.destroyed_time
        if resolved and drone.kill_type:
            if drone.kill_type == "ad":
                counts["ad"] += 1
            elif drone.kill_type == "interceptor":
                counts["interceptor"] += 1
            else:
                counts["ad"] += 1
        else:
            counts["active"] += 1
    counts["destroyed"] = len(series) - counts["active"]
    return counts


def _orientation_from_events(
    base_orientation: float,
    events: Sequence[Tuple[float, float]],
    t: float,
) -> float:
    orientation = base_orientation
    for event_time, direction in events:
        if t < event_time:
            break
        orientation = direction
    return orientation


def _build_animation(snapshot: Dict[str, object], output_path: Path, *, time_step: float, fps: int) -> None:
    series = _prepare_drone_series(snapshot)
    ad_units: Sequence[Dict[str, object]] = snapshot["ad_units"]  # type: ignore[assignment]
    target_events = _target_events(snapshot)
    max_t = _max_time(series)
    frames = int(math.ceil(max_t / time_step)) + 1

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_xlim(-1, ARENA_WIDTH + 1)
    ax.set_ylim(ARENA_HEIGHT + 1, -1)
    ax.set_title("Swarm Defense Large Animation")
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)

    target_base_markers = []
    target_kill_markers = []
    for idx, target in enumerate(target_events):
        base = ax.scatter(target["col"], target["row"], s=220, color="tab:green", marker="o", zorder=3)
        target_base_markers.append(base)
        kill = ax.scatter(
            target["col"],
            target["row"],
            s=260,
            marker=TARGET_KILL_MARKER,
            color=TARGET_KILL_COLOR,
            edgecolors=TARGET_KILL_EDGE,
            linewidths=1.6,
            alpha=0.0,
            zorder=6,
        )
        target_kill_markers.append(kill)
        ax.text(target["col"] + 0.3, target["row"], f"T{idx}\nV={target['value']:.0f}")

    arrow_length = AD_COVERAGE_RADIUS * 0.95
    ad_arrows = []
    for unit in ad_units:
        row, col = unit["position"]
        base_orientation = float(unit.get("orientation", math.pi / 2))
        raw_events: Sequence[Tuple[float, float]] = unit.get("orientation_events", ())
        events = sorted(raw_events, key=lambda item: item[0])
        arrow = patches.FancyArrowPatch(
            (col, row),
            (col, row),
            arrowstyle="->",
            color=AD_ROTATION_ARROW_COLOR,
            linewidth=1.6,
            mutation_scale=12,
        )
        ax.add_patch(arrow)
        ad_arrows.append({
            "arrow": arrow,
            "col": col,
            "row": row,
            "base": base_orientation,
            "events": events,
        })
        ax.scatter(col, row, s=220, marker="^", color="tab:blue")

    drone_markers = [ax.scatter(ser.entry[1], ser.entry[0], color=ser.color, s=50) for ser in series]
    path_lines = [ax.plot([], [], color=ser.color, linewidth=2)[0] for ser in series]
    kill_markers: List[Optional[object]] = []
    kill_pulses: List[Optional[patches.Circle]] = []
    interceptor_traces: List[Optional[Dict[str, object]]] = []
    interceptor_origin = (
        INTERCEPTOR_ORIGIN[0],
        min(max(INTERCEPTOR_ORIGIN[1], 0.0), ARENA_WIDTH),
    )
    for ser in series:
        if ser.kill_point is None:
            kill_markers.append(None)
            kill_pulses.append(None)
            interceptor_traces.append(None)
            continue
        row, col = ser.kill_point
        if ser.kill_type == "ad":
            marker = ax.scatter(
                col,
                row,
                facecolors=AD_KILL_COLOR,
                edgecolors=AD_KILL_EDGE,
                marker="X",
                s=150,
                linewidths=1.2,
                alpha=0.0,
                zorder=6,
            )
            pulse_color = AD_KILL_COLOR
        elif ser.kill_type == "interceptor":
            marker = ax.scatter(
                col,
                row,
                color=INTERCEPTOR_KILL_COLOR,
                marker="*",
                s=140,
                linewidths=1.2,
                edgecolors="black",
                alpha=0.0,
                zorder=6,
            )
            pulse_color = INTERCEPTOR_KILL_COLOR
        elif ser.kill_type == "ad_target":
            marker = ax.scatter(
                col,
                row,
                color=TARGET_KILL_COLOR,
                marker=TARGET_KILL_MARKER,
                s=150,
                linewidths=1.2,
                edgecolors=TARGET_KILL_EDGE,
                alpha=0.0,
                zorder=6,
            )
            pulse_color = TARGET_KILL_COLOR
        else:
            marker = ax.scatter(
                col,
                row,
                color=AD_TARGET_KILL_COLOR,
                marker="D",
                s=130,
                linewidths=1.0,
                edgecolors="black",
                alpha=0.0,
                zorder=6,
            )
            pulse_color = AD_TARGET_KILL_COLOR
        kill_markers.append(marker)
        pulse = patches.Circle(
            (col, row),
            radius=0.0,
            fill=False,
            linewidth=1.5,
            edgecolor=pulse_color,
            alpha=0.0,
            zorder=5,
        )
        ax.add_patch(pulse)
        kill_pulses.append(pulse)
        if ser.kill_type == "interceptor" and ser.destroyed_time is not None:
            marker = ax.scatter(
                interceptor_origin[1],
                interceptor_origin[0],
                color=INTERCEPTOR_KILL_COLOR,
                marker="^",
                s=70,
                alpha=0.0,
                zorder=7,
                linewidths=1.0,
                edgecolors="black",
            )
            track = ax.plot([], [], color=INTERCEPTOR_KILL_COLOR, linewidth=1.5, alpha=0.0)[0]
            start_time = max(0.0, ser.destroyed_time - INTERCEPTOR_VISUAL_DURATION)
            interceptor_traces.append(
                {
                    "marker": marker,
                    "track": track,
                    "start": start_time,
                    "end": ser.destroyed_time,
                    "kill_point": ser.kill_point,
                    "origin": interceptor_origin,
                }
            )
        else:
            interceptor_traces.append(None)
    status_text = ax.text(
        0.01,
        0.02,
        "",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )

    def _update(frame_idx: int):
        current_time = frame_idx * time_step
        counts = _status_counts(series, target_events, current_time)
        status_text.set_text(
            "t={time:.1f}s\nTargets destroyed: {targets}\nAD destroys: {ad}\n"
            "Intercepted: {intercepts}\nActive drones: {active}".format(
                time=current_time,
                targets=counts["targets"],
                ad=counts["ad"],
                intercepts=counts["interceptor"],
                active=counts["active"],
            )
        )
        for idx, target in enumerate(target_events):
            destroyed_time = target.get("destroyed_time")
            killed = destroyed_time is not None and current_time >= float(destroyed_time)
            target_base_markers[idx].set_alpha(0.0 if killed else 1.0)
            target_kill_markers[idx].set_alpha(1.0 if killed else 0.0)
        for idx, ser in enumerate(series):
            pos = _position_at(ser, current_time)
            drone_markers[idx].set_offsets([[pos[1], pos[0]]])
            prefix = _prefix_path(ser, current_time)
            xs = [p[1] for p in prefix]
            ys = [p[0] for p in prefix]
            path_lines[idx].set_data(xs, ys)
            if ser.destroyed_time is not None and current_time >= ser.destroyed_time:
                drone_markers[idx].set_alpha(0.4)
                drone_markers[idx].set_edgecolor("black")
                drone_markers[idx].set_facecolor("none")
            marker = kill_markers[idx]
            if marker is not None:
                if ser.destroyed_time is not None and current_time >= ser.destroyed_time:
                    marker.set_alpha(1.0)
                else:
                    marker.set_alpha(0.0)
            pulse = kill_pulses[idx]
            if pulse is not None and ser.destroyed_time is not None:
                if current_time < ser.destroyed_time:
                    pulse.set_alpha(0.0)
                    pulse.set_radius(0.0)
                else:
                    pulse_age = current_time - ser.destroyed_time
                    if pulse_age <= 1.5:
                        pulse.set_alpha(max(0.0, 1.0 - pulse_age / 1.5))
                        pulse.set_radius(0.2 + pulse_age * 0.8)
                    else:
                        pulse.set_alpha(0.0)
                        pulse.set_radius(0.0)
            elif pulse is not None:
                pulse.set_alpha(0.0)
                pulse.set_radius(0.0)
            trace = interceptor_traces[idx]
            if trace is not None:
                marker_artist = trace["marker"]
                track_artist = trace["track"]
                start = trace["start"]
                end = trace["end"]
                origin = trace["origin"]
                kill_point = trace["kill_point"]
                if current_time < start:
                    marker_artist.set_alpha(0.0)
                    track_artist.set_alpha(0.0)
                    track_artist.set_data([], [])
                elif current_time >= end:
                    marker_artist.set_alpha(0.0)
                    track_artist.set_alpha(0.0)
                    track_artist.set_data([], [])
                else:
                    duration = max(end - start, 1e-3)
                    progress = (current_time - start) / duration
                    eased = min(max(progress, 0.0), 1.0) ** 0.6
                    row = origin[0] + (kill_point[0] - origin[0]) * eased
                    col = origin[1] + (kill_point[1] - origin[1]) * eased
                    marker_artist.set_alpha(1.0)
                    marker_artist.set_offsets([[col, row]])
                    track_artist.set_data([origin[1], col], [origin[0], row])
                    track_artist.set_alpha(0.7)
        for data in ad_arrows:
            orientation = _orientation_from_events(data["base"], data["events"], current_time)
            end_col = data["col"] + arrow_length * math.cos(orientation)
            end_row = data["row"] + arrow_length * math.sin(orientation)
            data["arrow"].set_positions((data["col"], data["row"]), (end_col, end_row))
        artists = (
            drone_markers
            + path_lines
            + [data["arrow"] for data in ad_arrows]
            + [status_text]
        )
        artists.extend(target_base_markers)
        artists.extend(target_kill_markers)
        artists.extend(marker for marker in kill_markers if marker is not None)
        artists.extend(pulse for pulse in kill_pulses if pulse is not None)
        for trace in interceptor_traces:
            if trace is None:
                continue
            artists.append(trace["marker"])
            artists.append(trace["track"])
        return artists

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=frames,
        interval=1000 / fps,
        blit=False,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    anim.save(output_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Animate a Swarm Defense large episode with learned policy")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint directory with policy.pt")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed")
    parser.add_argument("--time-step", type=float, default=0.25, help="Simulation timestep in seconds")
    parser.add_argument("--fps", type=int, default=12, help="Frames per second for the GIF")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Output animation path")
    parser.add_argument(
        "--snapshot-output",
        type=Path,
        default=SNAPSHOT_OUTPUT_PATH,
        help="Where to store the JSON snapshot for WinTAK/CoT export",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    seed = args.seed if args.seed is not None else random.SystemRandom().randint(0, 2**31 - 1)
    
    print(f"Loading policy from: {checkpoint_path}")
    policy_net, metadata = load_policy_network(checkpoint_path)
    
    game_config = metadata['game_config']
    print(f"Policy trained on: {game_config['NUM_TARGETS']}T, "
          f"{game_config['NUM_AD_UNITS']}AD, {game_config['NUM_ATTACKING_DRONES']}D, "
          f"{game_config['NUM_INTERCEPTORS']}I")
    print(f"Small grid size: {game_config['GRID_SIZE']}×{game_config['GRID_SIZE']}")
    print(f"Large arena size: {ARENA_WIDTH:.0f}×{ARENA_HEIGHT:.0f}")
    print(f"Mapping: 1:1 (same agent counts, continuous positions)")
    print(f"Seed: {seed}\n")

    print("Running small game with policy...")
    small_snapshot = run_small_game_with_policy(policy_net, seed)
    
    print("Mapping to large game...")
    final_state = map_small_to_large_game(small_snapshot, seed)
    
    snapshot = final_state.snapshot()
    snapshot_path = _write_snapshot(snapshot, args.snapshot_output)
    
    print("Building animation...")
    _build_animation(snapshot, args.output, time_step=args.time_step, fps=args.fps)
    
    returns = final_state.returns()
    print("\nAnimation complete.")
    print(f"Seed: {seed}")
    print(f"Attacker damage: {returns[0]:.1f}")
    print(f"Defender reward: {returns[1]:.1f}")
    print(f"Snapshot saved to: {snapshot_path}")
    print(f"Animation saved to: {args.output}")


if __name__ == "__main__":
    main()
