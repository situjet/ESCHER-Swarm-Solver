"""Demo runner and visualizer for the Swarm Defense OpenSpiel game."""
from __future__ import annotations

import argparse
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


def _count_outcomes(drones: Tuple[Dict[str, object], ...]) -> Tuple[int, int, int]:
    ad = inter = surv = 0
    for drone in drones:
        destroyed_by = drone.get("destroyed_by") or ""
        if isinstance(destroyed_by, str) and destroyed_by.startswith("ad"):
            ad += 1
        elif isinstance(destroyed_by, str) and destroyed_by.startswith("interceptor"):
            inter += 1
        else:
            surv += 1
    return ad, inter, surv


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
    ad_kills, interceptor_kills, survivors = _count_outcomes(drones)

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

    for idx, target in enumerate(targets):
        ax.scatter(target.col, target.row, s=200, color="tab:green", marker="o")
        ax.text(target.col + 0.2, target.row + 0.2, f"T{idx}\nV={target.value}", color="black")

    for idx, unit in enumerate(ad_units):
        col, row = unit["position"][1], unit["position"][0]
        alive = bool(unit["alive"])
        color = "tab:blue" if alive else "tab:gray"
        marker = "^" if alive else "v"
        ax.scatter(col, row, marker=marker, s=200, color=color)
        status = "alive" if alive else f"KO ({unit.get('destroyed_by') or 'drone'})"
        ax.text(col - 0.6, row - 0.4, f"AD{idx}\n{status}", color=color, fontsize=8)

    intercept_label_added = False
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
        for ad_idx, intercept in drone["intercepts"]:
            hit_row, hit_col = intercept
            label = None
            if not intercept_label_added:
                label = "Intercept"
                intercept_label_added = True
            ax.scatter(
                hit_col,
                hit_row,
                color=color,
                marker="x",
                s=80,
                linewidths=2,
                label=label,
            )
            ax.text(
                hit_col + 0.1,
                hit_row + 0.1,
                f"AD{ad_idx}",
                color=color,
                fontsize=7,
            )

        ax.text(
            0.02,
            0.05,
            f"AD kills: {ad_kills}\nInterceptor kills: {interceptor_kills}\nSurvivors: {survivors}",
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
    ad_kills, interceptor_kills, survivors = _count_outcomes(state.snapshot()["drones"])
    print("Episode complete.")
    print(f"Seed used: {seed}")
    print(f"Attacker damage: {returns[0]:.1f}")
    print(f"Defender reward: {returns[1]:.1f}")
    print(
        f"Breakdown -> AD kills: {ad_kills}, Interceptor kills: {interceptor_kills}, Survivors: {survivors}"
    )
    print(f"Snapshot saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
