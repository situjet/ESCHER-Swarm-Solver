#!/usr/bin/env python3
"""Batch simulator to inspect AD and interceptor kill frequencies."""
from __future__ import annotations

import argparse
import random
from collections import Counter
from typing import Callable, Tuple

import pyspiel

import swarm_defense_game  # noqa: F401  # Registers the custom game.


def _count_outcomes(drones) -> Tuple[int, int, int]:
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


def _run_random_episode(seed: int):
    rng = random.Random(seed)
    game = pyspiel.load_game("swarm_defense")
    state = game.new_initial_state()
    while not state.is_terminal():
        player = state.current_player()
        if player == pyspiel.PlayerId.CHANCE:
            outcomes = state.chance_outcomes()
            pick = rng.random()
            cumulative = 0.0
            action = outcomes[-1][0]
            for act, prob in outcomes:
                cumulative += prob
                if pick <= cumulative:
                    action = act
                    break
        else:
            action = rng.choice(state.legal_actions())
        state.apply_action(action)
    return state


def _run_demo_episode(seed: int):
    from demo_visualizer import play_episode

    state, _ = play_episode(seed)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect kill-rate statistics over many episodes")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to roll out")
    parser.add_argument(
        "--policy",
        choices=("demo", "random"),
        default="demo",
        help="Which policy pair to use. demo=visualizer heuristics, random=uniform actions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base RNG seed; each episode uses seed+index for reproducibility.",
    )
    args = parser.parse_args()

    runner: Callable[[int], object]
    if args.policy == "demo":
        runner = _run_demo_episode
    else:
        runner = _run_random_episode

    totals = Counter()
    per_episode = []
    for i in range(args.episodes):
        state = runner(args.seed + i)
        ad, inter, surv = _count_outcomes(state.snapshot()["drones"])
        totals.update({"ad": ad, "interceptor": inter, "survivor": surv})
        per_episode.append((ad, inter, surv))

    drones_per_episode = sum(per_episode[0]) if per_episode else 0
    print("Episodes:", args.episodes)
    print("Policy:", args.policy)
    print("Totals:", dict(totals))
    if totals:
        total_drones = sum(totals.values())
        print(
            "Rates:",
            {
                "ad_pct": round(totals["ad"] / total_drones * 100, 2),
                "interceptor_pct": round(totals["interceptor"] / total_drones * 100, 2),
                "survivor_pct": round(totals["survivor"] / total_drones * 100, 2),
            },
        )
    if per_episode:
        avg_ad = sum(ad for ad, _, _ in per_episode) / args.episodes
        avg_inter = sum(inter for _, inter, _ in per_episode) / args.episodes
        avg_surv = sum(surv for _, _, surv in per_episode) / args.episodes
        print(
            "Per-episode averages:",
            {
                "ad": round(avg_ad, 2),
                "interceptor": round(avg_inter, 2),
                "survivor": round(avg_surv, 2),
                "drones": drones_per_episode,
            },
        )


if __name__ == "__main__":
    main()
