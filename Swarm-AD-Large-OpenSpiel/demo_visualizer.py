"""Demo runner and visualizer for the large Swarm Defense game."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import pyspiel

import policy_transfer
from policy_transfer import build_blueprint_from_small_snapshot, rollout_blueprint_episode
from swarm_defense_large_game import (
    AD_COVERAGE_RADIUS,
    AD_FOV_DEGREES,
    ARENA_HEIGHT,
    ARENA_WIDTH,
    LARGE_TOT_CHOICES,
    SwarmDefenseLargeState,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "Visualizer"
OUTPUT_PATH = OUTPUT_DIR / "swarm_defense_large_demo.png"
TOT_COLORS = {
    LARGE_TOT_CHOICES[0]: "#ff4d4f",
    LARGE_TOT_CHOICES[1]: "#ffa940",
    LARGE_TOT_CHOICES[2]: "#ffec3d",
    LARGE_TOT_CHOICES[3]: "#9254de",
}
DRONE_SPEED = 1.0
AD_KILL_COLOR = "#FF3B30"
AD_KILL_EDGE = "black"
AD_KILL_LINK = "#8C1B13"
INTERCEPTOR_KILL_COLOR = "#0bc5ea"
AD_TARGET_KILL_COLOR = "#2E8B57"


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
            legal = state.legal_actions()
            action = rng.choice(legal)
        state.apply_action(action)
    return state.snapshot()


def _count_outcomes(drones: Sequence[Dict[str, object]]) -> Tuple[int, int, int, int]:
    ad = inter = surv = ad_attrit = 0
    for drone in drones:
        destroyed = str(drone.get("destroyed_by") or "")
        if destroyed.startswith("ad_target"):
            ad_attrit += 1
        elif destroyed.startswith("ad"):
            ad += 1
        elif destroyed.startswith("interceptor"):
            inter += 1
        else:
            surv += 1
    return ad, inter, surv, ad_attrit


def _tot_color(tot: float) -> str:
    return TOT_COLORS.get(tot, "#ff4d4f")


def _kill_event(drone: Dict[str, object], ad_positions: Dict[int, Tuple[float, float]]) -> Optional[Dict[str, object]]:
    destroyed = str(drone.get("destroyed_by") or "")
    tot = float(drone.get("tot") or 0.0)
    intercepts: Sequence[Tuple[int, Tuple[float, float], float]] = drone.get("intercepts", ())
    if destroyed.startswith("ad:") and intercepts:
        ad_idx, hit_point, intercept_time = intercepts[0]
        kill_time = float(intercept_time)
        kill_distance = max(0.0, (kill_time - tot) * DRONE_SPEED)
        return {
            "type": "ad",
            "point": tuple(hit_point),
            "time": kill_time,
            "distance": kill_distance,
            "ad_idx": int(ad_idx),
        }
    if destroyed.startswith("interceptor"):
        hit = drone.get("interceptor_hit")
        kill_time = drone.get("interceptor_time")
        if hit is not None and kill_time is not None:
            kill_time = float(kill_time)
            kill_distance = max(0.0, (kill_time - tot) * DRONE_SPEED)
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
        kill_time = tot + total_distance / DRONE_SPEED
        return {
            "type": "ad_target",
            "point": kill_point,
            "time": kill_time,
            "distance": total_distance,
            "ad_idx": ad_idx,
        }
    return None


def _trim_samples(
    samples: Sequence[Tuple[float, float, float]],
    kill_distance: Optional[float],
) -> List[Tuple[float, float]]:
    if not samples:
        return []
    coords: List[Tuple[float, float]] = []
    eps = 1e-6
    prev_row, prev_col, prev_dist = samples[0]
    coords.append((prev_row, prev_col))
    if kill_distance is None:
        coords.extend((row, col) for row, col, _ in samples[1:])
        return coords
    for row, col, dist in samples[1:]:
        if dist < kill_distance - eps:
            coords.append((row, col))
            prev_row, prev_col, prev_dist = row, col, dist
            continue
        if dist <= kill_distance + eps:
            coords.append((row, col))
        else:
            if dist > prev_dist + eps:
                ratio = (kill_distance - prev_dist) / (dist - prev_dist)
                ratio = min(max(ratio, 0.0), 1.0)
                interp = (
                    prev_row + (row - prev_row) * ratio,
                    prev_col + (col - prev_col) * ratio,
                )
            else:
                interp = (row, col)
            coords.append(interp)
        break
    return coords


def _path_points(drone: Dict[str, object], kill_distance: Optional[float]) -> List[Tuple[float, float]]:
    samples: Sequence[Tuple[float, float, float]] = drone.get("path_samples", ())
    coords = _trim_samples(samples, kill_distance)
    if coords:
        return coords
    entry = tuple(drone["entry"])
    destination = tuple(drone.get("destination") or entry)
    return [entry, destination]


def _draw_kill_marker(
    ax: plt.Axes,
    kill: Dict[str, object],
    ad_positions: Dict[int, Tuple[float, float]],
) -> None:
    kill_row, kill_col = kill["point"]  # type: ignore[index]
    kill_time = kill.get("time")
    marker_type = kill.get("type")
    if marker_type == "ad":
        ad_idx = kill.get("ad_idx")
        if isinstance(ad_idx, int) and ad_idx in ad_positions:
            ad_row, ad_col = ad_positions[ad_idx]
            ax.plot(
                [ad_col, kill_col],
                [ad_row, kill_row],
                color=AD_KILL_LINK,
                linestyle=":",
                linewidth=1.3,
                alpha=0.9,
                zorder=3,
            )
        ax.scatter(
            kill_col,
            kill_row,
            facecolors=AD_KILL_COLOR,
            edgecolors=AD_KILL_EDGE,
            marker="X",
            s=180,
            linewidths=1.2,
            zorder=6,
        )
        if isinstance(kill_time, (int, float)):
            label = f"AD{kill.get('ad_idx', '?')}\nt={kill_time:.1f}"
            ax.text(
                kill_col + 0.2,
                kill_row + 0.15,
                label,
                fontsize=7,
                color="black",
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            )
    elif marker_type == "interceptor":
        ax.scatter(
            kill_col,
            kill_row,
            color=INTERCEPTOR_KILL_COLOR,
            marker="*",
            s=160,
            linewidths=1.2,
            edgecolors="black",
            zorder=6,
        )
        if isinstance(kill_time, (int, float)):
            ax.text(
                kill_col + 0.15,
                kill_row - 0.35,
                f"t={kill_time:.1f}",
                color=INTERCEPTOR_KILL_COLOR,
                fontsize=7,
            )
    elif marker_type == "ad_target":
        ax.scatter(
            kill_col,
            kill_row,
            color=AD_TARGET_KILL_COLOR,
            marker="D",
            s=150,
            linewidths=1.0,
            edgecolors="black",
            zorder=6,
        )
        ad_idx = kill.get("ad_idx")
        ax.text(
            kill_col + 0.15,
            kill_row + 0.15,
            f"AD{ad_idx if ad_idx is not None else '?'} neutralized",
            color=AD_TARGET_KILL_COLOR,
            fontsize=7,
        )


def _draw_ad_unit(ax, unit: Dict[str, object]) -> None:
    row, col = unit["position"]
    alive = bool(unit.get("alive", True))
    orientation = float(unit.get("orientation", 0.0))
    color = "tab:blue" if alive else "tab:gray"
    ax.scatter(col, row, s=220, marker="^" if alive else "v", color=color)
    wedge = patches.Wedge(
        center=(col, row),
        r=AD_COVERAGE_RADIUS,
        theta1=(orientation - AD_FOV_DEGREES / 2) * 180 / 3.14159,
        theta2=(orientation + AD_FOV_DEGREES / 2) * 180 / 3.14159,
        alpha=0.15,
        color="tab:blue",
    )
    ax.add_patch(wedge)


def render_snapshot(state: SwarmDefenseLargeState, output_path: Path) -> None:
    snapshot = state.snapshot()
    targets = snapshot["targets"]
    drones = snapshot["drones"]
    ad_units = snapshot["ad_units"]
    ad_kills, interceptor_kills, survivors, ad_attrit = _count_outcomes(drones)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_xlim(-1, ARENA_WIDTH + 1)
    ax.set_ylim(ARENA_HEIGHT + 1, -1)
    ax.set_title("Swarm Defense Large Snapshot")
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)

    for idx, target in enumerate(targets):
        row = target.row
        col = target.col
        ax.scatter(col, row, s=260, color="tab:green", marker="o")
        ax.text(col + 0.3, row, f"T{idx}\nV={target.value:.0f}", color="black")

    ad_positions: Dict[int, Tuple[float, float]] = {}
    for idx, unit in enumerate(ad_units):
        ad_positions[idx] = tuple(unit["position"])
        _draw_ad_unit(ax, unit)

    for idx, drone in enumerate(drones):
        entry_row, entry_col = drone["entry"]
        tot = float(drone.get("tot") or 0.0)
        color = _tot_color(tot)
        kill = _kill_event(drone, ad_positions)
        kill_distance = kill.get("distance") if isinstance(kill, dict) else None
        path_coords = _path_points(drone, kill_distance)
        xs = [point[1] for point in path_coords]
        ys = [point[0] for point in path_coords]
        ax.plot(xs, ys, color=color, linewidth=2, solid_capstyle="round", zorder=2)
        ax.scatter(
            entry_col,
            entry_row,
            color=color,
            marker="s",
            s=65,
            edgecolors="black",
            linewidths=0.6,
            zorder=4,
        )
        ax.text(entry_col - 0.5, entry_row - 0.5, f"D{idx}", color=color, fontsize=7, weight="bold")
        if kill is None:
            dest_row, dest_col = path_coords[-1]
            ax.scatter(
                dest_col,
                dest_row,
                facecolors="none",
                edgecolors=color,
                marker="o",
                s=130,
                linewidths=1.4,
                zorder=4,
            )
            ax.text(dest_col + 0.2, dest_row - 0.2, f"ToT={tot:.1f}", color=color, fontsize=7)
        else:
            _draw_kill_marker(ax, kill, ad_positions)

    legend_elements = [
        patches.Patch(facecolor="tab:green", edgecolor="black", label="Targets"),
        patches.Patch(facecolor="tab:blue", alpha=0.2, label="AD FOV"),
        mlines.Line2D([], [], color="black", marker="o", markerfacecolor="none", linestyle="None", label="Survivor dest"),
        mlines.Line2D([], [], color=AD_KILL_COLOR, marker="X", linestyle="None", label="AD intercept"),
        mlines.Line2D([], [], color=INTERCEPTOR_KILL_COLOR, marker="*", markeredgecolor="black", linestyle="None", label="Interceptor kill"),
        mlines.Line2D([], [], color=AD_TARGET_KILL_COLOR, marker="D", markeredgecolor="black", linestyle="None", label="AD-target strike"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    ax.text(
        0.01,
        0.04,
        f"AD intercepts: {ad_kills}\nAD-target strikes: {ad_attrit}\nInterceptor kills: {interceptor_kills}\nSurvivors: {survivors}",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.75},
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a large Swarm Defense demo episode")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.SystemRandom().randint(0, 2**31 - 1)
    rng = random.Random(seed)
    abstract_snapshot = _generate_abstract_snapshot(seed)
    blueprint = build_blueprint_from_small_snapshot(abstract_snapshot, rng=rng)
    final_state = rollout_blueprint_episode(blueprint, seed=seed)
    render_snapshot(final_state, OUTPUT_PATH)
    returns = final_state.returns()
    print("Large episode complete.")
    print(f"Seed: {seed}")
    print(f"Attacker damage: {returns[0]:.1f}")
    print(f"Defender reward: {returns[1]:.1f}")
    print(f"Snapshot saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
