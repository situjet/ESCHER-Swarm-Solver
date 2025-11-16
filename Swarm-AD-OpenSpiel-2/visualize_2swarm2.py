"""Generate a GIF visualizing a 2Swarm2 episode."""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib import animation
import pyspiel

from Swarm_AD_OpenSpiel_2 import two_swarm2_game as mod

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "2swarm2_demo.gif"
WAVE_COLORS = {1: "#ff6f61", 2: "#1f77b4"}
TIME_STEP = 0.5
FPS = 12
BUFFER_TIME = 4.0
WAVE_BREAK = 3.0
INTERCEPTOR_VISUAL_DURATION = 6.0


@dataclass
class DroneTrack:
    wave: int
    entry: Tuple[float, float]
    destination: Tuple[float, float]
    start_time: float
    end_time: float
    destroy_time: float
    kill_time: Optional[float]
    kill_point: Optional[Tuple[float, float]]
    kill_type: Optional[str]
    color: str
    interceptor_hit_time: Optional[float]
    interceptor_hit_point: Optional[Tuple[float, float]]


@dataclass
class InterceptorVisual:
    origin: Tuple[float, float]
    hit_point: Tuple[float, float]
    start_time: float
    hit_time: float


def _sample_chance_action(state: pyspiel.State, rng: random.Random) -> int:
    outcomes = state.chance_outcomes()
    pick = rng.random()
    cumulative = 0.0
    for action, probability in outcomes:
        cumulative += probability
        if pick <= cumulative:
            return action
    return outcomes[-1][0]


def _choose_drone_action(state: pyspiel.State, rng: random.Random) -> int:
    actions = state.legal_actions()
    snapshot = state.snapshot()
    current_wave = snapshot["current_wave"]
    assigned = sum(1 for drone in snapshot["drones"] if drone["wave"] == current_wave)
    if current_wave == 1:
        desired_wave1 = max(1, mod.NUM_ATTACKING_DRONES // 2)
        if assigned >= desired_wave1 and mod.DRONE_END_ACTION in actions:
            return mod.DRONE_END_ACTION
        filtered = [a for a in actions if a != mod.DRONE_END_ACTION]
        return rng.choice(filtered or actions)
    filtered = [a for a in actions if a != mod.DRONE_END_ACTION]
    return rng.choice(filtered or actions)


def _choose_intercept_action(state: pyspiel.State, rng: random.Random) -> int:
    actions = state.legal_actions()
    intercepts = [a for a in actions if a != mod.INTERCEPT_END_ACTION]
    if not intercepts:
        return mod.INTERCEPT_END_ACTION
    snapshot = state.snapshot()
    remaining = snapshot.get("remaining_interceptors", mod.NUM_INTERCEPTORS)
    current_wave = snapshot.get("current_wave", 1)
    waves_left = max(1, mod.NUM_WAVES - current_wave + 1)
    if remaining <= waves_left:
        # conserve ammo when approaching final waves
        trigger_probability = 0.5 if current_wave == mod.NUM_WAVES else 0.35
    else:
        trigger_probability = 0.75
    if rng.random() < trigger_probability:
        return rng.choice(intercepts)
    return mod.INTERCEPT_END_ACTION


def rollout_episode(seed: int) -> pyspiel.State:
    rng = random.Random(seed)
    game = mod.TwoSwarm2Game()
    state = game.new_initial_state()
    while not state.is_terminal():
        player = state.current_player()
        if player == pyspiel.PlayerId.CHANCE:
            state.apply_action(_sample_chance_action(state, rng))
            continue
        phase = state.phase()
        if phase == mod.Phase.SWARM_ASSIGNMENT:
            state.apply_action(_choose_drone_action(state, rng))
        elif phase == mod.Phase.INTERCEPT_ASSIGNMENT:
            state.apply_action(_choose_intercept_action(state, rng))
        else:
            state.apply_action(rng.choice(state.legal_actions()))
    return state


def _kill_event(drone: dict) -> Tuple[Optional[float], Optional[Tuple[float, float]], Optional[str]]:
    destroyed_by = str(drone.get("destroyed_by") or "")
    if destroyed_by.startswith("interceptor"):
        if drone.get("interceptor_time") is not None:
            return float(drone["interceptor_time"]), tuple(drone["interceptor_hit"]), "interceptor"
    if destroyed_by.startswith("ad:"):
        intercepts: Sequence[Tuple[int, Tuple[float, float], float]] = drone.get("intercepts", ())
        if intercepts:
            _, hit_point, intercept_time = intercepts[0]
            return float(intercept_time), tuple(hit_point), "ad"
    if destroyed_by.startswith("ad_target:"):
        return None, tuple(drone.get("destination")), "ad_target"
    if destroyed_by:
        return None, tuple(drone.get("destination")), destroyed_by
    if drone.get("strike_success"):
        return None, tuple(drone.get("destination")), "target"
    return None, None, None


def _wave_offsets(snapshot: dict) -> dict[int, float]:
    durations: dict[int, List[float]] = {}
    for drone in snapshot.get("drones", []):
        wave = int(drone.get("wave", 1))
        entry = tuple(drone["entry"])  # type: ignore[arg-type]
        destination = tuple(drone["destination"])  # type: ignore[arg-type]
        tot = float(drone.get("tot") or 0.0)
        distance = math.dist(entry, destination)
        durations.setdefault(wave, []).append(tot + distance)
    offsets: dict[int, float] = {}
    cumulative = 0.0
    for wave in sorted(durations.keys()):
        offsets[wave] = cumulative
        wave_length = max(durations[wave]) if durations[wave] else 0.0
        cumulative += wave_length + WAVE_BREAK
    return offsets


def build_tracks(snapshot: dict) -> List[DroneTrack]:
    offsets = _wave_offsets(snapshot)
    tracks: List[DroneTrack] = []
    for drone in snapshot.get("drones", []):
        entry = tuple(drone["entry"])  # type: ignore[arg-type]
        destination = tuple(drone["destination"])  # type: ignore[arg-type]
        tot = float(drone.get("tot") or 0.0)
        distance = math.dist(entry, destination)
        wave = int(drone.get("wave", 1))
        offset = offsets.get(wave, 0.0)
        start_time = offset + tot
        end_time = offset + tot + distance
        kill_time, kill_point, kill_type = _kill_event(drone)
        if kill_time is not None:
            kill_time += offset
        destroy_time = min(kill_time or end_time, end_time)
        interceptor_time = drone.get("interceptor_time")
        interceptor_hit = drone.get("interceptor_hit")
        if interceptor_time is not None:
            interceptor_time = float(interceptor_time) + offset
        if interceptor_hit is not None:
            interceptor_hit = tuple(interceptor_hit)
        color = WAVE_COLORS.get(drone.get("wave", 1), "#ff6f61")
        tracks.append(
            DroneTrack(
                wave=wave,
                entry=entry,
                destination=destination,
                start_time=start_time,
                end_time=end_time,
                destroy_time=destroy_time,
                kill_time=kill_time,
                kill_point=kill_point,
                kill_type=kill_type,
                color=color,
                interceptor_hit_time=interceptor_time,
                interceptor_hit_point=interceptor_hit,
            )
        )
    return tracks


def build_interceptor_visuals(tracks: Sequence[DroneTrack]) -> List[InterceptorVisual]:
    visuals: List[InterceptorVisual] = []
    for track in tracks:
        if track.interceptor_hit_time is None or track.interceptor_hit_point is None:
            continue
        origin = (14, 0)
        start_time = max(
            track.start_time,
            track.interceptor_hit_time - INTERCEPTOR_VISUAL_DURATION,
        )
        visuals.append(
            InterceptorVisual(
                origin=origin,
                hit_point=(track.interceptor_hit_point[0], track.interceptor_hit_point[1]),
                start_time=float(start_time),
                hit_time=float(track.interceptor_hit_time),
            )
        )
    return visuals


def animate(snapshot: dict, output_path: Path) -> None:
    tracks = build_tracks(snapshot)
    if not tracks:
        raise RuntimeError("No drones were deployed; cannot build animation")
    interceptor_visuals = build_interceptor_visuals(tracks)
    max_time = max(
        [track.destroy_time for track in tracks]
        + [vis.hit_time for vis in interceptor_visuals]
    ) + BUFFER_TIME
    frames = max(1, int(math.ceil(max_time / TIME_STEP)))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1, mod.GRID_SIZE)
    ax.set_ylim(mod.GRID_SIZE, -1)
    ax.set_xticks(range(0, mod.GRID_SIZE, 2))
    ax.set_yticks(range(0, mod.GRID_SIZE, 2))
    ax.set_title("2Swarm2 Episode Visualization")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.grid(True, linestyle="--", alpha=0.3)

    # Targets
    for idx, target in enumerate(snapshot.get("targets", [])):
        ax.scatter(target.col, target.row, marker="s", color="#2ecc71", s=120, alpha=0.8)
        ax.text(target.col + 0.2, target.row + 0.2, f"T{idx}", fontsize=8)

    # AD units with coverage
    for idx, unit in enumerate(snapshot.get("ad_units", [])):
        color = "#104e8b" if unit["alive"] else "#888888"
        ax.scatter(unit["position"][1], unit["position"][0], marker="^", s=140, color=color, zorder=3)
        circle = plt.Circle((unit["position"][1], unit["position"][0]), mod.AD_COVERAGE_RADIUS, fill=False, alpha=0.15)
        ax.add_patch(circle)
        ax.text(unit["position"][1] + 0.2, unit["position"][0] - 0.2, f"AD{idx}", fontsize=8)

    drone_paths = []
    drone_markers = []
    kill_markers = []
    kill_styles = {
        "interceptor": {"marker": "*", "color": "#0bc5ea", "size": 9},
        "ad": {"marker": "X", "color": "#FF3B30", "size": 9},
        "ad_target": {"marker": "X", "color": "#FF3B30", "size": 9},
        "target": {"marker": "P", "color": "#F1C40F", "size": 8},
    }
    default_kill_style = {"marker": "x", "color": "#ecf0f1", "size": 7}
    for track in tracks:
        path_line, = ax.plot([], [], color=track.color, linewidth=1.4, alpha=0.5)
        marker, = ax.plot([], [], marker="o", color=track.color, markersize=6, linestyle="")
        style = kill_styles.get(track.kill_type, default_kill_style)
        kill_marker, = ax.plot(
            [],
            [],
            marker=style["marker"],
            color=style["color"],
            markersize=style["size"],
            linestyle="",
            alpha=0.0,
        )
        drone_paths.append(path_line)
        drone_markers.append(marker)
        kill_markers.append(kill_marker)

    interceptor_artists: List[dict[str, object]] = []
    for vis in interceptor_visuals:
        line, = ax.plot([], [], color="#0bc5ea", linestyle="--", linewidth=1.2, alpha=0.0)
        marker, = ax.plot([], [], marker="*", color="#0bc5ea", markersize=9, linestyle="", alpha=0.0)
        pulse = plt.Circle((vis.hit_point[1], vis.hit_point[0]), radius=0.0, fill=False, linewidth=1.3, edgecolor="#0bc5ea", alpha=0.0)
        ax.add_patch(pulse)
        interceptor_artists.append({"visual": vis, "line": line, "marker": marker, "pulse": pulse})

    wave_windows = {}
    for wave in sorted({track.wave for track in tracks}):
        wave_tracks = [t for t in tracks if t.wave == wave]
        if not wave_tracks:
            continue
        start = min(t.start_time for t in wave_tracks)
        end = max(t.end_time for t in wave_tracks)
        wave_windows[wave] = (start, end)

    status_text = ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=11, color="#222222")

    def _interpolate(track: DroneTrack, current_time: float) -> Tuple[float, float]:
        if current_time <= track.start_time or track.end_time <= track.start_time:
            return track.entry
        if current_time >= track.destroy_time:
            if track.kill_point is not None:
                # kill_point stored as (row, col)
                return (track.kill_point[0], track.kill_point[1])
            return track.destination
        ratio = (current_time - track.start_time) / (track.end_time - track.start_time)
        ratio = max(0.0, min(1.0, ratio))
        row = track.entry[0] + (track.destination[0] - track.entry[0]) * ratio
        col = track.entry[1] + (track.destination[1] - track.entry[1]) * ratio
        return (row, col)

    def init() -> List[object]:
        for path_line, marker, kill_marker in zip(drone_paths, drone_markers, kill_markers):
            path_line.set_data([], [])
            marker.set_data([], [])
            kill_marker.set_data([], [])
            kill_marker.set_alpha(0.0)
        for intercept in interceptor_artists:
            line = intercept["line"]
            marker = intercept["marker"]
            pulse = intercept["pulse"]
            line.set_data([], [])
            line.set_alpha(0.0)
            marker.set_data([], [])
            marker.set_alpha(0.0)
            pulse.set_alpha(0.0)
            pulse.set_radius(0.0)
        status_text.set_text("")
        artists: List[object] = drone_paths + drone_markers + kill_markers
        artists += [intercept["line"] for intercept in interceptor_artists]
        artists += [intercept["marker"] for intercept in interceptor_artists]
        artists += [intercept["pulse"] for intercept in interceptor_artists]
        artists.append(status_text)
        return artists

    def _current_wave_label(current_time: float) -> int:
        ordered = sorted(wave_windows.items())
        for wave, (start, end) in ordered:
            if start <= current_time <= end + WAVE_BREAK * 0.5:
                return wave
        return ordered[-1][0] if ordered else 1

    def update(frame: int) -> List[object]:
        current_time = frame * TIME_STEP
        wave_label = _current_wave_label(current_time)
        status_text.set_text(f"Wave {wave_label}  |  t = {current_time:.1f}s")
        artists: List[object] = [status_text]
        for track, path_line, marker, kill_marker in zip(tracks, drone_paths, drone_markers, kill_markers):
            if current_time < track.start_time:
                path_line.set_data([], [])
                marker.set_data([], [])
                marker.set_alpha(0.0)
                kill_marker.set_alpha(0.0)
                artists.extend([path_line, marker, kill_marker])
                continue
            marker.set_alpha(1.0)
            row, col = _interpolate(track, current_time)
            path_line.set_data([track.entry[1], col], [track.entry[0], row])
            marker.set_data([col], [row])
            if track.kill_time is not None and current_time >= track.kill_time and track.kill_point is not None:
                kill_marker.set_data([track.kill_point[1]], [track.kill_point[0]])
                kill_marker.set_alpha(0.9)
            else:
                kill_marker.set_alpha(0.0)
            artists.extend([path_line, marker, kill_marker])
        for intercept in interceptor_artists:
            vis: InterceptorVisual = intercept["visual"]
            line = intercept["line"]
            marker_artist = intercept["marker"]
            pulse = intercept["pulse"]
            if current_time < vis.start_time:
                line.set_alpha(0.0)
                line.set_data([], [])
                marker_artist.set_alpha(0.0)
                marker_artist.set_data([], [])
                pulse.set_alpha(0.0)
                pulse.set_radius(0.0)
            elif current_time <= vis.hit_time:
                duration = max(vis.hit_time - vis.start_time, 1e-6)
                progress = (current_time - vis.start_time) / duration
                eased = min(max(progress, 0.0), 1.0) ** 0.7
                row = vis.origin[0] + (vis.hit_point[0] - vis.origin[0]) * eased
                col = vis.origin[1] + (vis.hit_point[1] - vis.origin[1]) * eased
                line.set_data([vis.origin[1], col], [vis.origin[0], row])
                line.set_alpha(0.65)
                marker_artist.set_data([col], [row])
                marker_artist.set_alpha(1.0)
                pulse.set_alpha(0.0)
                pulse.set_radius(0.0)
            elif current_time <= vis.hit_time + 1.0:
                age = current_time - vis.hit_time
                fade = max(0.0, 1.0 - age / 1.0)
                line.set_alpha(0.0)
                line.set_data([], [])
                marker_artist.set_alpha(0.0)
                marker_artist.set_data([], [])
                pulse.set_center((vis.hit_point[1], vis.hit_point[0]))
                pulse.set_radius(0.15 + age * 0.8)
                pulse.set_alpha(fade)
            else:
                line.set_alpha(0.0)
                line.set_data([], [])
                marker_artist.set_alpha(0.0)
                marker_artist.set_data([], [])
                pulse.set_alpha(0.0)
                pulse.set_radius(0.0)
            artists.extend([line, marker_artist, pulse])
        return artists

    ani = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=frames,
        interval=1000 / FPS,
        blit=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(output_path, writer=animation.PillowWriter(fps=FPS))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a 2Swarm2 GIF")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for the rollout")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="GIF output path")
    args = parser.parse_args()

    final_state = rollout_episode(args.seed)
    animate(final_state.snapshot(), args.output)
    print(f"Saved 2Swarm2 animation to {args.output}")


if __name__ == "__main__":
    main()
