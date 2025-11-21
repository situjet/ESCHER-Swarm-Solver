"""Heuristic rollout helpers that force two-wave behavior for the v2 demo scripts."""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import pyspiel

from swarm_defense_large_game import (
    DRONE_ALLOC_ACTION_BASE,
    INTERCEPT_END_ACTION,
    INTERCEPT_ALLOC_ACTION_BASE,
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
    if not isinstance(state, SwarmDefenseLargeState):
        return rng.choice(legal)

    phase = state.phase()
    if phase == Phase.SWARM_WAVE_ALLOCATION:
        return _attacker_wave_allocation_action(state, rng, wave_budgets)
    if phase != Phase.SWARM_ASSIGNMENT:
        return rng.choice(legal)

    snapshot = state.snapshot()
    current_wave = int(snapshot.get("current_wave", 1))
    remaining = int(snapshot.get("remaining_drones", 0))
    drones: Sequence[Dict[str, object]] = snapshot.get("drones", ())  # type: ignore[assignment]
    budget = wave_budgets.get(current_wave)
    if budget is None:
        budget = _new_wave_budget(current_wave, remaining, rng)
        wave_budgets[current_wave] = budget
    drone_actions = list(legal)
    if not drone_actions:
        return rng.choice(legal)
    return _prioritize_drone_action(drone_actions, snapshot, rng)


def _attacker_wave_allocation_action(
    state: SwarmDefenseLargeState,
    rng: random.Random,
    wave_budgets: Dict[int, int],
) -> int:
    legal = state.legal_actions()
    if not legal:
        raise RuntimeError("No legal attacker allocation actions available")
    snapshot = state.snapshot()
    current_wave = int(snapshot.get("current_wave", 1))
    remaining = int(snapshot.get("remaining_drones", 0))
    budget = wave_budgets.get(current_wave)
    if budget is None:
        budget = _new_wave_budget(current_wave, remaining, rng)
    desired = remaining if current_wave >= NUM_WAVES else budget
    desired = max(0, min(desired, remaining))
    allocations = sorted(action - DRONE_ALLOC_ACTION_BASE for action in legal)
    if not allocations:
        return rng.choice(legal)
    desired = max(allocations[0], min(desired, allocations[-1]))
    wave_budgets[current_wave] = desired
    action = DRONE_ALLOC_ACTION_BASE + desired
    if action not in legal:
        fallback = DRONE_ALLOC_ACTION_BASE + allocations[-1]
        return fallback if fallback in legal else rng.choice(legal)
    return action


def _new_wave_budget(current_wave: int, remaining: int, rng: random.Random) -> int:
    if current_wave >= NUM_WAVES:
        return max(1, remaining)
    if remaining <= 1:
        return 1
    wave_min = max(1, int(remaining * 0.25))
    wave_max = max(wave_min, int(remaining * 0.75))
    budget = rng.randint(wave_min, wave_max)
    if remaining - budget < 1:
        budget = max(1, remaining - 1)
    return max(1, min(budget, remaining))


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
    if not isinstance(state, SwarmDefenseLargeState):
        return rng.choice(legal)

    phase = state.phase()
    if phase == Phase.INTERCEPTOR_WAVE_ALLOCATION:
        return _defender_wave_allocation_action(state, rng, interceptor_caps, interceptors_fired)
    if phase != Phase.INTERCEPT_ASSIGNMENT:
        return rng.choice(legal)

    snapshot = state.snapshot()
    current_wave = int(snapshot.get("current_wave", 1))
    intercept_actions = [action for action in legal if action != INTERCEPT_END_ACTION]
    cap = interceptor_caps.get(current_wave)
    if cap is None:
        cap = int(snapshot.get("wave_interceptors_active", 0))
        interceptor_caps[current_wave] = cap
    fired = interceptors_fired.get(current_wave, 0)

    if cap <= 0 or fired >= cap or not intercept_actions:
        return INTERCEPT_END_ACTION if INTERCEPT_END_ACTION in legal else rng.choice(legal)

    action = _prioritize_interceptor_action(intercept_actions, snapshot, rng)
    interceptors_fired[current_wave] = fired + 1
    return action


def _defender_wave_allocation_action(
    state: SwarmDefenseLargeState,
    rng: random.Random,
    interceptor_caps: Dict[int, int],
    interceptors_fired: Dict[int, int],
) -> int:
    legal = state.legal_actions()
    if not legal:
        raise RuntimeError("No legal interceptor allocation actions available")
    snapshot = state.snapshot()
    current_wave = int(snapshot.get("current_wave", 1))
    remaining = int(snapshot.get("remaining_interceptors", 0))
    cap = interceptor_caps.get(current_wave)
    if cap is None:
        base = remaining if current_wave >= NUM_WAVES else max(0, int(NUM_INTERCEPTORS * 0.55))
        cap = min(base, remaining)
    cap = max(0, min(cap, remaining))
    allocations = sorted(action - INTERCEPT_ALLOC_ACTION_BASE for action in legal)
    if not allocations:
        return rng.choice(legal)
    cap = max(allocations[0], min(cap, allocations[-1]))
    interceptor_caps[current_wave] = cap
    interceptors_fired[current_wave] = 0
    action = INTERCEPT_ALLOC_ACTION_BASE + cap
    if action not in legal:
        fallback = INTERCEPT_ALLOC_ACTION_BASE + allocations[-1]
        return fallback if fallback in legal else rng.choice(legal)
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
