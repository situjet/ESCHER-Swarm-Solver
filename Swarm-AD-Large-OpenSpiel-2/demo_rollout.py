"""Heuristic rollout helpers that force two-wave behavior for the v2 demo scripts."""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import pyspiel

from swarm_defense_large_game import (
    DRONE_END_ACTION,
    INTERCEPT_END_ACTION,
    NUM_ATTACKING_DRONES,
    NUM_INTERCEPTORS,
    NUM_WAVES,
    Phase,
    SwarmDefenseLargeGame,
    SwarmDefenseLargeState,
    decode_drone_action,
    decode_interceptor_action,
)

__all__ = ["rollout_two_wave_episode"]


def rollout_two_wave_episode(
    seed: Optional[int], *, require_ad_discovery: bool = True
) -> SwarmDefenseLargeState:
    """Generate a full episode with scripted wave splitting.

    The attacker deliberately holds drones for later waves and the defender
    meters interceptor usage so both waves produce visible activity in the
    visualizer/animation outputs.
    """

    rng = random.Random(seed)
    params = {}
    if not require_ad_discovery:
        params["require_ad_discovery"] = 0
    state = SwarmDefenseLargeGame(params).new_initial_state()
    wave_budgets: Dict[int, int] = {}
    interceptor_caps: Dict[int, int] = {}
    interceptors_fired: Dict[int, int] = {}

    while not state.is_terminal():
        player = state.current_player()
        if player == pyspiel.PlayerId.CHANCE:
            action = _sample_chance_action(state, rng)
        elif player == 0:
            action = _attacker_action(state, rng, wave_budgets)
        else:
            action = _defender_action(state, rng, interceptor_caps, interceptors_fired)
        state.apply_action(action)
    return state


def _sample_chance_action(state: pyspiel.State, rng: random.Random) -> int:
    outcomes = state.chance_outcomes()
    pick = rng.random()
    cumulative = 0.0
    for action, probability in outcomes:
        cumulative += probability
        if pick <= cumulative:
            return action
    return outcomes[-1][0]


def _attacker_action(
    state: pyspiel.State,
    rng: random.Random,
    wave_budgets: Dict[int, int],
) -> int:
    legal = state.legal_actions()
    if not legal:
        raise RuntimeError("No legal attacker actions available")
    if not isinstance(state, SwarmDefenseLargeState) or state.phase() != Phase.SWARM_ASSIGNMENT:
        return rng.choice(legal)

    snapshot = state.snapshot()
    current_wave = int(snapshot.get("current_wave", 1))
    remaining = int(snapshot.get("remaining_drones", 0))
    drones: Sequence[Dict[str, object]] = snapshot.get("drones", ())  # type: ignore[assignment]
    launched = sum(1 for drone in drones if int(drone.get("wave", 1)) == current_wave)
    budget = wave_budgets.get(current_wave)
    if budget is None:
        budget = _new_wave_budget(current_wave, remaining, rng)
        wave_budgets[current_wave] = budget

    if current_wave < NUM_WAVES and DRONE_END_ACTION in legal and launched >= budget:
        return DRONE_END_ACTION

    drone_actions = [action for action in legal if action != DRONE_END_ACTION]
    if not drone_actions:
        return DRONE_END_ACTION if DRONE_END_ACTION in legal else rng.choice(legal)
    return _prioritize_drone_action(drone_actions, snapshot, rng)


def _new_wave_budget(current_wave: int, remaining: int, rng: random.Random) -> int:
    if current_wave >= NUM_WAVES:
        return max(1, remaining)
    if remaining <= 1:
        return 1
    reserve_min = max(2, NUM_ATTACKING_DRONES // 3)
    reserve_max = max(reserve_min + 1, NUM_ATTACKING_DRONES // 2)
    reserve = rng.randint(reserve_min, reserve_max)
    reserve = min(reserve, max(0, remaining - 1))
    budget = max(1, remaining - reserve)
    return min(budget, remaining)


def _prioritize_drone_action(
    actions: Sequence[int],
    snapshot: Dict[str, object],
    rng: random.Random,
) -> int:
    targets: Sequence[object] = snapshot.get("targets", ())  # type: ignore[assignment]
    discovered_ads = set(snapshot.get("discovered_ads", ()))
    total_targets = len(targets)
    current_wave = int(snapshot.get("current_wave", 1))
    drones: Sequence[Dict[str, object]] = snapshot.get("drones", ())  # type: ignore[assignment]
    wave_assignments: Dict[int, int] = {}
    for drone in drones:
        if int(drone.get("wave", 1)) != current_wave:
            continue
        target_idx = int(drone.get("target_idx", -1))
        if target_idx < 0:
            continue
        wave_assignments[target_idx] = wave_assignments.get(target_idx, 0) + 1
    scored: List[Tuple[Tuple[float, float, float], int]] = []
    for action in actions:
        entry_idx, target_idx, tot_idx, blueprint_idx = decode_drone_action(action)
        if target_idx < total_targets:
            value = getattr(targets[target_idx], "value", 0.0)
            load = wave_assignments.get(target_idx, 0)
            score = (float(load), -float(value), rng.random())
        else:
            ad_idx = target_idx - total_targets
            discovered = 1.0 if ad_idx in discovered_ads else 0.0
            load = wave_assignments.get(target_idx, 0)
            score = (discovered + float(load), float(ad_idx), rng.random())
        scored.append((score, action))
    scored.sort(key=lambda item: item[0])
    top_k = max(1, len(scored) // 5)
    candidates = [action for _, action in scored[:top_k]]
    return rng.choice(candidates)


def _defender_action(
    state: pyspiel.State,
    rng: random.Random,
    interceptor_caps: Dict[int, int],
    interceptors_fired: Dict[int, int],
) -> int:
    legal = state.legal_actions()
    if not legal:
        raise RuntimeError("No legal defender actions available")
    if not isinstance(state, SwarmDefenseLargeState) or state.phase() != Phase.INTERCEPT_ASSIGNMENT:
        return rng.choice(legal)

    snapshot = state.snapshot()
    current_wave = int(snapshot.get("current_wave", 1))
    remaining = int(snapshot.get("remaining_interceptors", 0))
    intercept_actions = [action for action in legal if action != INTERCEPT_END_ACTION]
    cap = interceptor_caps.get(current_wave)
    if cap is None:
        cap = remaining if current_wave >= NUM_WAVES else max(1, int(NUM_INTERCEPTORS * 0.55))
        interceptor_caps[current_wave] = cap
    fired = interceptors_fired.get(current_wave, 0)

    if remaining <= 0 or fired >= cap or not intercept_actions:
        return INTERCEPT_END_ACTION if INTERCEPT_END_ACTION in legal else rng.choice(legal)

    action = _prioritize_interceptor_action(intercept_actions, snapshot, rng)
    interceptors_fired[current_wave] = fired + 1
    return action


def _prioritize_interceptor_action(
    actions: Sequence[int],
    snapshot: Dict[str, object],
    rng: random.Random,
) -> int:
    drones: Sequence[Dict[str, object]] = snapshot.get("drones", ())  # type: ignore[assignment]
    targets: Sequence[object] = snapshot.get("targets", ())  # type: ignore[assignment]
    total_targets = len(targets)
    scored: List[Tuple[Tuple[float, float, float], int]] = []

    for action in actions:
        drone_idx = decode_interceptor_action(action)
        if not (0 <= drone_idx < len(drones)):
            scored.append(((2.0, rng.random(), 0.0), action))
            continue
        drone = drones[drone_idx]
        destroyed = str(drone.get("destroyed_by") or "")
        if destroyed:
            scored.append(((1.5, rng.random(), 0.0), action))
            continue
        target_idx = int(drone.get("target_idx", -1))
        if 0 <= target_idx < total_targets:
            value = getattr(targets[target_idx], "value", 0.0)
            score = (0.0, -float(value), rng.random())
        else:
            # Treat AD-targeting drones as very high priority once legal.
            score = (0.0, -80.0, rng.random())
        scored.append((score, action))
    scored.sort(key=lambda item: item[0])
    top_k = max(1, len(scored) // 3)
    candidates = [action for _, action in scored[:top_k]]
    return rng.choice(candidates)
