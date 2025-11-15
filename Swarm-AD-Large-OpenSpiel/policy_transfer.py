"""Transfer utilities between the abstract and large Swarm-AD games."""
from __future__ import annotations

import importlib
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pyspiel
from open_spiel.python import policy as os_policy

try:  # Allow direct script usage without package context.
    from .swarm_defense_large_game import (
        ENTRY_POINTS,
        LARGE_TOT_CHOICES,
        MIDPOINT_STRATEGIES,
        NUM_ATTACKING_DRONES,
        Phase,
        SwarmDefenseLargeGame,
        SwarmDefenseLargeState,
        encode_drone_action,
    )
except ImportError:  # pragma: no cover - fallback for script execution
    from swarm_defense_large_game import (  # type: ignore
        ENTRY_POINTS,
        LARGE_TOT_CHOICES,
        MIDPOINT_STRATEGIES,
        NUM_ATTACKING_DRONES,
        Phase,
        SwarmDefenseLargeGame,
        SwarmDefenseLargeState,
        encode_drone_action,
    )

_ABSTRACT_MODULE_NAME = "swarm_defense_game"


def _load_abstract_module():
    if _ABSTRACT_MODULE_NAME in sys.modules:
        return sys.modules[_ABSTRACT_MODULE_NAME]
    repo_root = Path(__file__).resolve().parents[1]
    module_dir = repo_root / "Swarm-AD-OpenSpiel"
    module_dir_str = str(module_dir)
    if module_dir_str not in sys.path:
        sys.path.insert(0, module_dir_str)
    return importlib.import_module(_ABSTRACT_MODULE_NAME)


abstract_game = _load_abstract_module()

GridPoint = Tuple[float, float]


@dataclass
class DroneBlueprintAssignment:
    """Represents a single lifted drone assignment."""

    entry_lane_idx: int
    target_idx: int
    tot_idx: int
    blueprint_idx: int
    priority: float = 1.0


class BlueprintStrategy:
    """Container for a set of drone assignments executable in the large game."""

    def __init__(self, assignments: Sequence[DroneBlueprintAssignment]):
        if not assignments:
            raise ValueError("Blueprint requires at least one assignment")
        self.assignments: List[DroneBlueprintAssignment] = sorted(
            assignments,
            key=lambda assignment: assignment.priority,
            reverse=True,
        )

    def top_k(self, count: int) -> "BlueprintStrategy":
        return BlueprintStrategy(self.assignments[:count])

    def __len__(self) -> int:  # pragma: no cover - helper
        return len(self.assignments)

    def iter_actions(self) -> Iterable[int]:
        for assignment in self.assignments:
            yield encode_drone_action(
                assignment.entry_lane_idx,
                assignment.target_idx,
                assignment.tot_idx,
                assignment.blueprint_idx,
            )

    def apply(self, state: SwarmDefenseLargeState) -> None:
        if state.phase() != Phase.SWARM_ASSIGNMENT:
            raise ValueError("State must be in SWARM_ASSIGNMENT phase before applying blueprint")
        for action in self.iter_actions():
            if state.phase() != Phase.SWARM_ASSIGNMENT:
                break
            state.apply_action(action)

    @classmethod
    def merge(
        cls, strategies: Sequence["BlueprintStrategy"], *, max_assignments: Optional[int] = None
    ) -> "BlueprintStrategy":
        merged: List[DroneBlueprintAssignment] = []
        for strategy in strategies:
            merged.extend(strategy.assignments)
        if not merged:
            raise ValueError("Cannot merge empty blueprint collection")
        merged.sort(key=lambda assignment: assignment.priority, reverse=True)
        if max_assignments is not None:
            merged = merged[:max_assignments]
        return cls(merged)


def _map_entry_lane(entry_col: float, copy_idx: int, copies: int, rng: random.Random) -> int:
    ratio = entry_col / max(1, abstract_game.GRID_SIZE - 1)
    base_lane = round(ratio * (len(ENTRY_POINTS) - 1))
    spread = [-2, -1, 0, 1]
    offset = spread[copy_idx % len(spread)]
    jitter = rng.choice([-1, 0, 1]) if copies > len(spread) else 0
    lane = base_lane + offset + jitter
    lane = max(0, min(len(ENTRY_POINTS) - 1, lane))
    return lane


def _map_tot_value(tot_value: float) -> int:
    best_idx = 0
    best_delta = float("inf")
    for idx, option in enumerate(LARGE_TOT_CHOICES):
        delta = abs(option - tot_value)
        if delta < best_delta:
            best_idx = idx
            best_delta = delta
    return best_idx


def _select_blueprint_name(
    entry: GridPoint,
    destination: GridPoint,
    target_is_ad: bool,
    copy_idx: int,
) -> str:
    if target_is_ad:
        return "loiter"
    lateral_delta = destination[1] - entry[1]
    if lateral_delta < -1.5:
        return "fan_left"
    if lateral_delta > 1.5:
        return "fan_right"
    if copy_idx % 4 == 3:
        return "loiter"
    return "direct"


def _blueprint_index(name: str) -> int:
    return MIDPOINT_STRATEGIES.index(name)


def _target_priority(drone_info: Dict[str, object], copy_idx: int) -> float:
    base = float(drone_info.get("target_value") or 0.0)
    tot_penalty = float(drone_info.get("tot") or 0.0) * 0.2
    return max(0.1, base - tot_penalty - copy_idx * 0.5)


def build_blueprint_from_small_snapshot(
    snapshot: Dict[str, object],
    *,
    scale_factor: int = 4,
    rng: Optional[random.Random] = None,
) -> BlueprintStrategy:
    if rng is None:
        rng = random.Random()
    drones: Sequence[Dict[str, object]] = snapshot.get("drones", ())  # type: ignore[assignment]
    if not drones:
        raise ValueError("Snapshot does not contain drone assignments")
    num_targets = len(snapshot.get("targets", ()))
    assignments: List[DroneBlueprintAssignment] = []
    for drone in drones:
        entry: GridPoint = tuple(drone.get("entry", (0.0, 0.0)))  # type: ignore[arg-type]
        target_idx = int(drone.get("target_idx", 0))
        destination: GridPoint = tuple(drone.get("destination", entry))  # type: ignore[arg-type]
        tot_value = float(drone.get("tot") or 0.0)
        target_is_ad = target_idx >= num_targets
        priority = _target_priority(drone, 0)
        for copy_idx in range(scale_factor):
            lane = _map_entry_lane(entry[1], copy_idx, scale_factor, rng)
            blueprint_name = _select_blueprint_name(entry, destination, target_is_ad, copy_idx)
            blueprint_idx = _blueprint_index(blueprint_name)
            tot_idx = _map_tot_value(tot_value)
            adjusted_priority = priority - copy_idx * 0.25
            assignments.append(
                DroneBlueprintAssignment(
                    entry_lane_idx=lane,
                    target_idx=target_idx,
                    tot_idx=tot_idx,
                    blueprint_idx=blueprint_idx,
                    priority=adjusted_priority,
                )
            )
    assignments.sort(key=lambda assignment: assignment.priority, reverse=True)
    assignments = assignments[:NUM_ATTACKING_DRONES]
    return BlueprintStrategy(assignments)


def _sample_chance_action(state: pyspiel.State, rng: random.Random) -> int:
    outcomes = state.chance_outcomes()
    pick = rng.random()
    cumulative = 0.0
    for action, probability in outcomes:
        cumulative += probability
        if pick <= cumulative:
            return action
    return outcomes[-1][0]


def _sample_policy_action(
    solver_policy: os_policy.Policy,
    state: pyspiel.State,
    player_id: int,
    rng: random.Random,
) -> int:
    action_probs = solver_policy.action_probabilities(state, player_id)
    if not action_probs:
        return rng.choice(state.legal_actions(player_id))
    actions, probs = zip(*action_probs.items())
    return rng.choices(actions, weights=probs, k=1)[0]


def lift_policy_to_blueprint(
    solver_policy: os_policy.Policy,
    *,
    num_samples: int = 4,
    seed: Optional[int] = None,
    scale_factor: int = 4,
) -> BlueprintStrategy:
    rng = random.Random(seed)
    strategies: List[BlueprintStrategy] = []
    abstract_game_instance = abstract_game.SwarmDefenseGame()
    for _ in range(max(1, num_samples)):
        state = abstract_game_instance.new_initial_state()
        while not state.is_terminal():
            current_player = state.current_player()
            if current_player == pyspiel.PlayerId.CHANCE:
                action = _sample_chance_action(state, rng)
            else:
                action = _sample_policy_action(solver_policy, state, current_player, rng)
            state.apply_action(action)
        snapshot = state.snapshot()
        blueprint = build_blueprint_from_small_snapshot(snapshot, scale_factor=scale_factor, rng=rng)
        strategies.append(blueprint)
    return BlueprintStrategy.merge(strategies, max_assignments=NUM_ATTACKING_DRONES)


def apply_blueprint_to_large_state(state: SwarmDefenseLargeState, blueprint: BlueprintStrategy) -> None:
    blueprint.apply(state)


def rollout_blueprint_episode(
    blueprint: BlueprintStrategy,
    *,
    defender_policy: Optional[os_policy.Policy] = None,
    seed: Optional[int] = None,
) -> SwarmDefenseLargeState:
    rng = random.Random(seed)
    game = SwarmDefenseLargeGame()
    state = game.new_initial_state()
    while state.phase() == Phase.TARGET_POSITIONS:
        action = _sample_chance_action(state, rng)
        state.apply_action(action)
    while state.phase() == Phase.TARGET_VALUES:
        action = _sample_chance_action(state, rng)
        state.apply_action(action)
    while state.phase() == Phase.AD_PLACEMENT:
        legal = state.legal_actions()
        state.apply_action(rng.choice(legal))
        if state.phase() != Phase.AD_PLACEMENT:
            break
    if state.phase() != Phase.SWARM_ASSIGNMENT:
        raise RuntimeError("Unexpected phase before swarm assignment")
    apply_blueprint_to_large_state(state, blueprint)
    while not state.is_terminal():
        player = state.current_player()
        if player == pyspiel.PlayerId.CHANCE:
            action = _sample_chance_action(state, rng)
        elif player == 1 and defender_policy is not None:
            action = _sample_policy_action(defender_policy, state, player, rng)
        else:
            legal = state.legal_actions()
            action = rng.choice(legal)
        state.apply_action(action)
    return state


__all__ = [
    "DroneBlueprintAssignment",
    "BlueprintStrategy",
    "build_blueprint_from_small_snapshot",
    "lift_policy_to_blueprint",
    "apply_blueprint_to_large_state",
    "rollout_blueprint_episode",
]
