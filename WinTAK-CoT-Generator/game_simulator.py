"""Game state simulator for Swarm Defense visualization."""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Add parent directories to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "Swarm-AD-Large-OpenSpiel"))

try:
    import pyspiel
    import policy_transfer
    from swarm_defense_large_game import (
        AD_COVERAGE_RADIUS,
        ARENA_HEIGHT,
        ARENA_WIDTH,
        DRONE_SPEED,
        INTERCEPTOR_SPEED_MULTIPLIER,
        LARGE_TOT_CHOICES,
    )
except ImportError:
    # Fallback if imports fail
    ARENA_WIDTH = 32.0
    ARENA_HEIGHT = 32.0
    AD_COVERAGE_RADIUS = 5.5
    LARGE_TOT_CHOICES = (0.0, 1.5, 3.0, 4.5)
    DRONE_SPEED = 1.0
    INTERCEPTOR_SPEED_MULTIPLIER = 2.0

INTERCEPT_BACKLINE_ROW = ARENA_HEIGHT + 2.0
INTERCEPTOR_SPEED = max(DRONE_SPEED * INTERCEPTOR_SPEED_MULTIPLIER, 1.0)
AD_FLASH_WINDOW = 1.2  # seconds of flashing connection prior to intercept


@dataclass
class DroneState:
    """State of a single drone at a point in time."""

    drone_id: int
    position: Tuple[float, float]
    status: str  # "active", "destroyed", "completed"
    target_idx: int
    tot_offset: float
    path_points: List[Tuple[float, float]]
    path_times: List[float]
    destroyed_by: Optional[str] = None
    kill_point: Optional[Tuple[float, float]] = None
    kill_time: Optional[float] = None
    completion_time: Optional[float] = None
    intercepts: List[Tuple[int, Tuple[float, float], float]] = field(default_factory=list)
    strike_success: bool = False
    visible: bool = True


@dataclass
class InterceptorTrack:
    """Planned path for an interceptor engagement."""

    interceptor_id: int
    path_points: List[Tuple[float, float]]
    path_times: List[float]
    target_drone_id: int
    start_time: float
    end_time: float


@dataclass
class InterceptorState:
    """State of an interceptor at a point in time."""

    interceptor_id: int
    position: Tuple[float, float]
    assigned_drone: Optional[int]
    visible: bool = True


@dataclass
class ADUnitState:
    """State of an air defense unit."""

    ad_id: int
    position: Tuple[float, float]
    orientation: float
    coverage_radius: float = AD_COVERAGE_RADIUS
    engaged_drones: List[int] = field(default_factory=list)
    alive: bool = True
    destroyed_by: Optional[str] = None


@dataclass
class TargetState:
    """State of a target cluster."""

    target_id: int
    position: Tuple[float, float]
    value: float
    tot_offset: float
    destroy_time: Optional[float] = None
    is_destroyed: bool = False
    visible: bool = True


@dataclass
class ADEngagement:
    """Active AD engagement for flashing link rendering."""

    ad_id: int
    drone_id: int
    ad_position: Tuple[float, float]
    drone_position: Tuple[float, float]
    intercept_time: float


@dataclass
class GameSnapshot:
    """Complete game state at a single time step."""

    time: float
    drones: List[DroneState]
    interceptors: List[InterceptorState]
    ad_units: List[ADUnitState]
    targets: List[TargetState]
    engagements: List[ADEngagement]


def _sample_chance_action(state, rng: random.Random) -> int:
    """Sample a chance action from the state."""

    outcomes = state.chance_outcomes()
    pick = rng.random()
    cumulative = 0.0
    for action, probability in outcomes:
        cumulative += probability
        if pick <= cumulative:
            return action
    return outcomes[-1][0]


def _generate_abstract_snapshot(seed: int) -> Dict[str, object]:
    """Generate an abstract game snapshot using the small game."""

    rng = random.Random(seed)
    try:
        abstract_game = policy_transfer.abstract_game.SwarmDefenseGame()
        state = abstract_game.new_initial_state()
        while not state.is_terminal():
            player = state.current_player()
            if player == pyspiel.PlayerId.CHANCE:
                action = _sample_chance_action(state, rng)
            else:
                action = rng.choice(state.legal_actions())
            state.apply_action(action)
        return state.snapshot()
    except Exception as exc:  # pragma: no cover - fallback path
        print(f"Warning: Could not generate abstract snapshot: {exc}")
        return {}


def _build_large_game_state(seed: int) -> Optional[object]:
    """Build a complete large game state."""

    try:
        rng = random.Random(seed)
        abstract_snapshot = _generate_abstract_snapshot(seed)
        if not abstract_snapshot:
            return None
        blueprint = policy_transfer.build_blueprint_from_small_snapshot(abstract_snapshot, rng=rng)
        final_state = policy_transfer.rollout_blueprint_episode(blueprint, seed=seed)
        return final_state
    except Exception as exc:  # pragma: no cover - fallback path
        print(f"Warning: Could not build large game state: {exc}")
        return None


def _interpolate_position(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    ratio: float,
) -> Tuple[float, float]:
    """Linearly interpolate between two positions."""

    ratio = max(0.0, min(1.0, ratio))
    return (
        p1[0] + (p2[0] - p1[0]) * ratio,
        p1[1] + (p2[1] - p1[1]) * ratio,
    )


def _position_on_path(path_points: Sequence[Tuple[float, float]], path_times: Sequence[float], t: float) -> Tuple[float, float]:
    """Generic helper to compute a position along a timed path."""

    if not path_points:
        return (0.0, 0.0)
    if not path_times or t <= path_times[0]:
        return path_points[0]

    prev_time = path_times[0]
    prev_point = path_points[0]
    for next_time, next_point in zip(path_times[1:], path_points[1:]):
        if t <= next_time:
            if next_time == prev_time:
                return next_point
            ratio = (t - prev_time) / max(next_time - prev_time, 1e-6)
            return _interpolate_position(prev_point, next_point, ratio)
        prev_time = next_time
        prev_point = next_point
    return path_points[-1]


def _build_interceptor_tracks(drones: List[DroneState]) -> List[InterceptorTrack]:
    """Create interceptor tracks for drones killed by interceptors."""

    tracks: List[InterceptorTrack] = []
    counter = 0
    for drone in drones:
        if not drone.destroyed_by or not str(drone.destroyed_by).startswith("interceptor"):
            continue
        if drone.kill_point is None or drone.kill_time is None:
            continue
        entry_col = drone.path_points[0][1] if drone.path_points else drone.kill_point[1]
        start_point = (INTERCEPT_BACKLINE_ROW, entry_col)
        distance = math.dist(start_point, drone.kill_point)
        flight_time = max(distance / INTERCEPTOR_SPEED, 0.5)
        start_time = max(0.0, drone.kill_time - flight_time)
        tracks.append(
            InterceptorTrack(
                interceptor_id=counter,
                path_points=[start_point, drone.kill_point],
                path_times=[start_time, drone.kill_time],
                target_drone_id=drone.drone_id,
                start_time=start_time,
                end_time=drone.kill_time,
            )
        )
        counter += 1
    return tracks


def _compute_target_destroy_times(drones: List[DroneState]) -> Dict[int, float]:
    destroy_times: Dict[int, float] = {}
    for drone in drones:
        if not drone.strike_success or not drone.path_times:
            continue
        arrival = drone.path_times[-1]
        current = destroy_times.get(drone.target_idx)
        destroy_times[drone.target_idx] = arrival if current is None else min(current, arrival)
    return destroy_times


def _convert_snapshot_to_game_states(game_state_obj, time_step: float = 0.25) -> List[GameSnapshot]:
    """Convert a game state object to a sequence of GameSnapshot objects."""

    if game_state_obj is None:
        return _generate_synthetic_game_states(time_step)

    try:
        snapshot = game_state_obj.snapshot()
        return _parse_large_game_snapshot(snapshot, time_step)
    except Exception as exc:
        print(f"Warning: Could not parse game snapshot: {exc}")
        return _generate_synthetic_game_states(time_step)


def _parse_large_game_snapshot(snapshot: Dict, time_step: float) -> List[GameSnapshot]:
    """Parse a large game snapshot into time-stepped game states."""

    drones_data: Sequence[Dict] = snapshot.get("drones", [])
    ad_units_data: Sequence[Dict] = snapshot.get("ad_units", [])
    targets_data = snapshot.get("targets", [])

    drone_tracks: List[DroneState] = []
    for idx, drone_data in enumerate(drones_data):
        entry = tuple(drone_data["entry"])
        tot = float(drone_data.get("tot") or 0.0)
        hold = float(drone_data.get("hold_time") or 0.0)
        samples: Sequence[Tuple[float, float, float]] = drone_data.get("path_samples", [])
        path_points = [entry]
        path_times = [hold]
        for row, col, dist in samples:
            path_points.append((row, col))
            path_times.append(hold + dist / DRONE_SPEED)

        destroyed_by = drone_data.get("destroyed_by")
        kill_point = None
        kill_time = None
        if destroyed_by:
            intercepts = drone_data.get("intercepts", [])
            if str(destroyed_by).startswith("ad:") and intercepts:
                _, hit_point, intercept_time = intercepts[0]
                kill_point = tuple(hit_point)
                kill_time = float(intercept_time)
            elif str(destroyed_by).startswith("interceptor"):
                hit = drone_data.get("interceptor_hit")
                kill_point = tuple(hit) if hit else None
                kill_time = float(drone_data.get("interceptor_time") or 0.0) if hit else None

        strike_success = bool(drone_data.get("strike_success"))
        completion_time = path_times[-1] if (strike_success and path_times) else None
        intercepts_raw: Sequence[Tuple[int, Tuple[float, float], float]] = drone_data.get("intercepts", [])
        intercepts = [(int(ad_idx), tuple(point), float(time_val)) for ad_idx, point, time_val in intercepts_raw]

        drone_tracks.append(
            DroneState(
                drone_id=idx,
                position=entry,
                status="active",
                target_idx=drone_data.get("target_idx", 0),
                tot_offset=tot,
                path_points=path_points,
                path_times=path_times,
                destroyed_by=destroyed_by,
                kill_point=kill_point,
                kill_time=kill_time,
                completion_time=completion_time,
                intercepts=intercepts,
                strike_success=strike_success,
            )
        )

    ad_units_template: List[ADUnitState] = []
    for idx, ad_data in enumerate(ad_units_data):
        pos = tuple(ad_data["position"])
        orientation = float(ad_data.get("orientation", math.pi / 2))
        alive = bool(ad_data.get("alive", True))
        ad_units_template.append(
            ADUnitState(
                ad_id=idx,
                position=pos,
                orientation=orientation,
                alive=alive,
                destroyed_by=ad_data.get("destroyed_by"),
            )
        )

    target_destroy_times = _compute_target_destroy_times(drone_tracks)
    target_destroyed_flags = snapshot.get("target_destroyed", [])

    targets_template: List[TargetState] = []
    for idx, target_data in enumerate(targets_data):
        if hasattr(target_data, "row"):
            pos = (target_data.row, target_data.col)
            value = float(target_data.value)
        else:
            pos = (target_data["row"], target_data["col"])
            value = float(target_data.get("value", 0))
        destroy_time = target_destroy_times.get(idx)
        flags_set = target_destroyed_flags[idx] if idx < len(target_destroyed_flags) else False
        targets_template.append(
            TargetState(
                target_id=idx,
                position=pos,
                value=value,
                tot_offset=0.0,
                destroy_time=destroy_time,
                is_destroyed=flags_set,
            )
        )

    interceptor_tracks = _build_interceptor_tracks(drone_tracks)

    max_time = 0.0
    for drone in drone_tracks:
        if drone.path_times:
            max_time = max(max_time, drone.path_times[-1])
        if drone.kill_time:
            max_time = max(max_time, drone.kill_time + 1.0)
    max_time += 2.0

    snapshots: List[GameSnapshot] = []
    t = 0.0
    while t <= max_time:
        current_drones: List[DroneState] = []
        engagements: List[ADEngagement] = []
        snapshot_ad_units: List[ADUnitState] = [
            ADUnitState(
                ad_id=ad.ad_id,
                position=ad.position,
                orientation=ad.orientation,
                coverage_radius=ad.coverage_radius,
                alive=ad.alive,
                destroyed_by=ad.destroyed_by,
            )
            for ad in ad_units_template
        ]

        for track in drone_tracks:
            position = _position_on_path(track.path_points, track.path_times, t)
            status = "active"
            visible = True
            if track.kill_time is not None and t >= track.kill_time:
                status = "destroyed"
                visible = False
            elif track.strike_success and track.completion_time is not None:
                if t >= track.completion_time:
                    status = "completed"
                    visible = False
            current_drones.append(
                DroneState(
                    drone_id=track.drone_id,
                    position=position,
                    status=status,
                    target_idx=track.target_idx,
                    tot_offset=track.tot_offset,
                    path_points=track.path_points,
                    path_times=track.path_times,
                    destroyed_by=track.destroyed_by,
                    kill_point=track.kill_point,
                    kill_time=track.kill_time,
                    completion_time=track.completion_time,
                    intercepts=track.intercepts,
                    strike_success=track.strike_success,
                    visible=visible,
                )
            )

        for track, drone_state in zip(drone_tracks, current_drones):
            for ad_idx, _hit_point, intercept_time in track.intercepts:
                if intercept_time is None:
                    continue
                if t > intercept_time or t < intercept_time - AD_FLASH_WINDOW:
                    continue
                if ad_idx >= len(snapshot_ad_units):
                    continue
                snapshot_ad_units[ad_idx].engaged_drones.append(track.drone_id)
                engagements.append(
                    ADEngagement(
                        ad_id=ad_idx,
                        drone_id=track.drone_id,
                        ad_position=snapshot_ad_units[ad_idx].position,
                        drone_position=drone_state.position,
                        intercept_time=intercept_time,
                    )
                )

        current_interceptors: List[InterceptorState] = []
        for track in interceptor_tracks:
            if not (track.start_time <= t < track.end_time):
                continue
            position = _position_on_path(track.path_points, track.path_times, t)
            current_interceptors.append(
                InterceptorState(
                    interceptor_id=track.interceptor_id,
                    position=position,
                    assigned_drone=track.target_drone_id,
                    visible=True,
                )
            )

        current_targets: List[TargetState] = []
        for target in targets_template:
            visible = True
            destroy_time = target.destroy_time
            if destroy_time is not None and t >= destroy_time:
                visible = False
            current_targets.append(
                TargetState(
                    target_id=target.target_id,
                    position=target.position,
                    value=target.value,
                    tot_offset=target.tot_offset,
                    destroy_time=target.destroy_time,
                    is_destroyed=target.is_destroyed,
                    visible=visible,
                )
            )

        snapshots.append(
            GameSnapshot(
                time=t,
                drones=current_drones,
                interceptors=current_interceptors,
                ad_units=snapshot_ad_units,
                targets=current_targets,
                engagements=engagements,
            )
        )
        t += time_step

    return snapshots


def _generate_synthetic_game_states(time_step: float = 0.25) -> List[GameSnapshot]:
    """Generate synthetic game states for testing when real game data unavailable."""

    print("Generating synthetic game states for testing...")
    targets_template = [
        TargetState(0, (20.0, 8.0), 40.0, 0.0),
        TargetState(1, (22.0, 16.0), 25.0, 0.0),
        TargetState(2, (24.0, 24.0), 10.0, 0.0),
    ]
    ad_units_template = [
        ADUnitState(0, (18.0, 10.0), math.pi / 2),
        ADUnitState(1, (20.0, 20.0), math.pi / 2),
    ]

    drones: List[DroneState] = []
    for i in range(12):
        entry_col = (i * 2.5) % 24
        entry = (0.0, entry_col)
        target_idx = i % len(targets_template)
        target_pos = targets_template[target_idx].position
        tot = LARGE_TOT_CHOICES[i % len(LARGE_TOT_CHOICES)]
        distance = math.dist(entry, target_pos)
        travel_time = distance / DRONE_SPEED
        path_points = [entry, target_pos]
        path_times = [tot, tot + travel_time]
        destroyed_by = None
        kill_point = None
        kill_time = None
        intercepts: List[Tuple[int, Tuple[float, float], float]] = []
        strike_success = i % 4 == 0
        if i % 5 == 0:
            destroyed_by = "interceptor"
            kill_point = target_pos
            kill_time = tot + travel_time * 0.6
        elif i % 3 == 0:
            destroyed_by = "ad:0"
            kill_point = target_pos
            kill_time = tot + travel_time * 0.7
            intercepts.append((0, target_pos, kill_time))
        drones.append(
            DroneState(
                drone_id=i,
                position=entry,
                status="active",
                target_idx=target_idx,
                tot_offset=tot,
                path_points=path_points,
                path_times=path_times,
                destroyed_by=destroyed_by,
                kill_point=kill_point,
                kill_time=kill_time,
                completion_time=path_times[-1] if strike_success else None,
                intercepts=intercepts,
                strike_success=strike_success,
            )
        )

    synthetic_snapshot = {
        "drones": [],
        "ad_units": [
            {"position": ad.position, "orientation": ad.orientation, "alive": ad.alive}
            for ad in ad_units_template
        ],
        "targets": [
            {"row": tgt.position[0], "col": tgt.position[1], "value": tgt.value}
            for tgt in targets_template
        ],
        "target_destroyed": [False for _ in targets_template],
    }
    # Inject drone-like dicts expected by parser
    for drone in drones:
        synthetic_snapshot["drones"].append(
            {
                "entry": drone.path_points[0],
                "tot": drone.tot_offset,
                "hold_time": drone.path_times[0],
                "path_samples": [
                    (point[0], point[1], max(0.0, (t - drone.path_times[0]) * DRONE_SPEED))
                    for point, t in zip(drone.path_points[1:], drone.path_times[1:])
                ],
                "destroyed_by": drone.destroyed_by,
                "interceptor_hit": drone.kill_point,
                "interceptor_time": drone.kill_time,
                "target_idx": drone.target_idx,
                "strike_success": drone.strike_success,
                "intercepts": drone.intercepts,
            }
        )
    return _parse_large_game_snapshot(synthetic_snapshot, time_step)


def generate_game_sequence(seed: int, time_step: float = 0.25) -> List[GameSnapshot]:
    """Generate a complete sequence of game states."""

    game_state = _build_large_game_state(seed)
    return _convert_snapshot_to_game_states(game_state, time_step)
