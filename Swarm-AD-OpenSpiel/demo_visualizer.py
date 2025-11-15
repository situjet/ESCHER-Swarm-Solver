"""Demo runner and visualizer for the Swarm Defense OpenSpiel game."""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyspiel

from swarm_defense_game import (
    AD_POSITION_CANDIDATES,
    Phase,
    SwarmDefenseState,
    TOT_CHOICES,
)

GRID_SIZE = 16
BOTTOM_HALF_START = GRID_SIZE // 2
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "Visualizer"
OUTPUT_PATH = OUTPUT_DIR / "swarm_defense_demo.png"

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


def play_episode(seed: Optional[int] = None) -> Tuple[SwarmDefenseState, int]:
    if seed is None:
        seed = random.SystemRandom().randint(0, 2**31 - 1)
    rng = random.Random(seed)
    import swarm_defense_game  # noqa: F401  # Registers the custom game.

    game = pyspiel.load_game("swarm_defense")
    state = game.new_initial_state()
    while not state.is_terminal():
        player = state.current_player()
        if player == pyspiel.PlayerId.CHANCE:
            action = _sample_chance_action(state, rng)
        elif player == 0:
            action = _attacker_policy(state, rng)
        else:
            action = _defender_policy(state, rng)
        state.apply_action(action)
    return state, seed


def render_snapshot(state: SwarmDefenseState, output_path: Path) -> None:
    snapshot = state.snapshot()
    targets = snapshot["targets"]
    drones = snapshot["drones"]
    ad_units = snapshot["ad_units"]
    target_statuses = _compute_target_kill_status(drones, targets)
    ad_kills, interceptor_kills, survivors, ad_attrit = _count_outcomes(drones)

    fig, ax = plt.subplots(figsize=(8, 8))
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    ax.imshow(grid, cmap="Greys", alpha=0.05, extent=(-0.5, GRID_SIZE - 0.5, GRID_SIZE - 0.5, -0.5))
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    ax.grid(True, which="both", color="lightgrey", linewidth=0.5, alpha=0.5)
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(GRID_SIZE - 0.5, -0.5)
    ax.set_title("Swarm Defense Episode Snapshot")

    ax.add_patch(
        plt.Rectangle(
            (-0.5, BOTTOM_HALF_START - 0.5),
            GRID_SIZE,
            GRID_SIZE - BOTTOM_HALF_START,
            linewidth=1.0,
            edgecolor="tab:blue",
            facecolor="none",
            linestyle="--",
            label="Bottom-half AO",
        )
    )
    # highlight even-stride candidate cells
    for row, col in AD_POSITION_CANDIDATES:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and render a Swarm Defense episode")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducible runs")
    args = parser.parse_args()

    state, seed = play_episode(args.seed)
    render_snapshot(state, OUTPUT_PATH)
    returns = state.returns()
    ad_kills, interceptor_kills, survivors, ad_attrit = _count_outcomes(state.snapshot()["drones"])
    print("Episode complete.")
    print(f"Seed used: {seed}")
    print(f"Attacker damage: {returns[0]:.1f}")
    print(f"Defender reward: {returns[1]:.1f}")
    print(
        "Breakdown -> AD kills: {ad} (intercepts), AD-target strikes: {attrit}, "
        "Interceptor kills: {inter}, Survivors: {surv}".format(
            ad=ad_kills, attrit=ad_attrit, inter=interceptor_kills, surv=survivors
        )
    )
    print(f"Snapshot saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
