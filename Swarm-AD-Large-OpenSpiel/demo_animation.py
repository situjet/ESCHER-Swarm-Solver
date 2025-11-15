"""Time-stepped animation for the large Swarm Defense scenario."""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib import animation, patches
import pyspiel

import policy_transfer
from policy_transfer import build_blueprint_from_small_snapshot, rollout_blueprint_episode
from swarm_defense_large_game import (
    AD_COVERAGE_RADIUS,
    ARENA_HEIGHT,
    ARENA_WIDTH,
    LARGE_TOT_CHOICES,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "Visualizer"
OUTPUT_PATH = OUTPUT_DIR / "swarm_defense_large_animation.gif"
DRONE_SPEED = 1.0
AD_KILL_COLOR = "#FF3B30"
AD_KILL_EDGE = "black"
INTERCEPTOR_KILL_COLOR = "#0bc5ea"
AD_TARGET_KILL_COLOR = "#2E8B57"
AD_ROTATION_ARROW_COLOR = "#1f4c94"


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


def _sample_chance_action(state: pyspiel.State, rng: random.Random) -> int:
    outcomes = state.chance_outcomes()
    pick = rng.random()
    cumulative = 0.0
    for action, probability in outcomes:
        cumulative += probability
        if pick <= cumulative:
            return action
    return outcomes[-1][0]


def _generate_abstract_snapshot(seed: int) -> Dict[str, object]:
    rng = random.Random(seed)
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


def _orientation_from_events(
    base_orientation: float,
    events: Sequence[Tuple[float, float, float, float]],
    t: float,
) -> float:
    orientation = base_orientation
    for start_time, end_time, start_dir, end_dir in events:
        if t < start_time:
            break
        if end_time > start_time and start_time <= t <= end_time:
            span = max(end_time - start_time, 1e-6)
            ratio = min(max((t - start_time) / span, 0.0), 1.0)
            return _interpolate_angle(start_dir, end_dir, ratio)
        if t >= end_time:
            orientation = end_dir
    return orientation


def _build_animation(state: policy_transfer.SwarmDefenseLargeState, output_path: Path, *, time_step: float, fps: int) -> None:
    snapshot = state.snapshot()
    series = _prepare_drone_series(snapshot)
    ad_units: Sequence[Dict[str, object]] = snapshot["ad_units"]  # type: ignore[assignment]
    targets: Sequence[Dict[str, object]] = snapshot["targets"]  # type: ignore[assignment]
    max_t = _max_time(series)
    frames = int(math.ceil(max_t / time_step)) + 1

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_xlim(-1, ARENA_WIDTH + 1)
    ax.set_ylim(ARENA_HEIGHT + 1, -1)
    ax.set_title("Swarm Defense Large Animation")
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)

    for idx, target in enumerate(targets):
        ax.scatter(target.col, target.row, s=220, color="tab:green", marker="o")
        ax.text(target.col + 0.3, target.row, f"T{idx}\nV={target.value:.0f}")

    arrow_length = AD_COVERAGE_RADIUS * 0.95
    ad_arrows = []
    for unit in ad_units:
        row, col = unit["position"]
        base_orientation = float(unit.get("orientation", math.pi / 2))
        events: Sequence[Tuple[float, float, float, float]] = unit.get("orientation_events", ())
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
    for ser in series:
        if ser.kill_point is None:
            kill_markers.append(None)
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
        kill_markers.append(marker)
    time_text = ax.text(0.01, 0.02, "t=0.0", transform=ax.transAxes)

    def _update(frame_idx: int):
        current_time = frame_idx * time_step
        time_text.set_text(f"t={current_time:.1f}")
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
        for data in ad_arrows:
            orientation = _orientation_from_events(data["base"], data["events"], current_time)
            end_col = data["col"] + arrow_length * math.cos(orientation)
            end_row = data["row"] + arrow_length * math.sin(orientation)
            data["arrow"].set_positions((data["col"], data["row"]), (end_col, end_row))
        artists = drone_markers + path_lines + [data["arrow"] for data in ad_arrows] + [time_text]
        artists.extend(marker for marker in kill_markers if marker is not None)
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
    parser = argparse.ArgumentParser(description="Animate a Swarm Defense large episode")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed")
    parser.add_argument("--time-step", type=float, default=0.25, help="Simulation timestep in seconds")
    parser.add_argument("--fps", type=int, default=12, help="Frames per second for the GIF")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Output animation path")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.SystemRandom().randint(0, 2**31 - 1)
    rng = random.Random(seed)
    abstract_snapshot = _generate_abstract_snapshot(seed)
    blueprint = build_blueprint_from_small_snapshot(abstract_snapshot, rng=rng)
    final_state = rollout_blueprint_episode(blueprint, seed=seed)
    _build_animation(final_state, args.output, time_step=args.time_step, fps=args.fps)
    print("Animation complete.")
    print(f"Seed: {seed}")
    print(f"Animation saved to: {args.output}")


if __name__ == "__main__":
    main()
