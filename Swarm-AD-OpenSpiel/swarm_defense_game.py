"""Custom OpenSpiel game modeling a swarm attack vs. layered air defense."""
from __future__ import annotations

import itertools
import math
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence, Tuple

import pyspiel

GRID_SIZE = 16
BOTTOM_HALF_START = GRID_SIZE // 2
NUM_TARGETS = 3
NUM_AD_UNITS = 2
NUM_ATTACKING_DRONES = 10
NUM_INTERCEPTORS = 3
TOT_CHOICES: Tuple[float, ...] = (0.0, 2.0, 4.0)
TARGET_VALUE_OPTIONS: Tuple[float, ...] = (10.0, 20.0, 40.0)
AD_COVERAGE_RADIUS = 5.0
AD_STRIDE = 2


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


AD_KILL_RATE = _env_float("SWARM_AD_KILL_RATE", 0.9)  # per unit distance inside bubble

TARGET_CANDIDATE_CELLS: List[Tuple[int, int]] = [
    (row, col)
    for row in range(BOTTOM_HALF_START, GRID_SIZE)
    for col in range(GRID_SIZE)
]
TARGET_VALUE_PERMUTATIONS: List[Sequence[int]] = list(
    itertools.permutations(range(NUM_TARGETS))
)

AD_POSITION_CANDIDATES: List[Tuple[int, int]] = [
    (row, col)
    for row in range(BOTTOM_HALF_START, GRID_SIZE)
    if row % AD_STRIDE == 0
    for col in range(GRID_SIZE)
    if col % AD_STRIDE == 0
]

DRONE_TARGET_SLOTS = NUM_TARGETS + NUM_AD_UNITS

TARGET_POSITION_ACTIONS = len(TARGET_CANDIDATE_CELLS)
TARGET_VALUE_ACTIONS = len(TARGET_VALUE_PERMUTATIONS)
AD_PLACEMENT_ACTIONS = len(AD_POSITION_CANDIDATES)
DRONE_ASSIGNMENT_ACTIONS = GRID_SIZE * DRONE_TARGET_SLOTS * len(TOT_CHOICES)
INTERCEPT_CHOICES = NUM_ATTACKING_DRONES + 1

TARGET_POSITION_ACTION_BASE = 0
TARGET_VALUE_ACTION_BASE = TARGET_POSITION_ACTION_BASE + TARGET_POSITION_ACTIONS
AD_ACTION_BASE = TARGET_VALUE_ACTION_BASE + TARGET_VALUE_ACTIONS
DRONE_ACTION_BASE = AD_ACTION_BASE + AD_PLACEMENT_ACTIONS
INTERCEPT_ACTION_BASE = DRONE_ACTION_BASE + DRONE_ASSIGNMENT_ACTIONS
AD_RESOLVE_ACTION_BASE = INTERCEPT_ACTION_BASE + INTERCEPT_CHOICES
NUM_DISTINCT_ACTIONS = AD_RESOLVE_ACTION_BASE + 2


@dataclass(frozen=True)
class TargetCluster:
    row: int
    col: int
    value: float


@dataclass
class ADUnit:
    row: int
    col: int
    alive: bool = True
    destroyed_by: Optional[str] = None
    intercept_log: List[Tuple[int, Tuple[float, float]]] = field(default_factory=list)


@dataclass
class DronePlan:
    entry_row: int
    entry_col: int
    target_idx: int
    tot_idx: int
    destroyed_by: Optional[str] = None
    intercepts: List[Tuple[int, Tuple[float, float]]] = field(default_factory=list)


@dataclass
class ADIntercept:
    ad_idx: int
    drone_idx: int
    exposure: float
    probability: float
    hit_point: Tuple[float, float]


class Phase(Enum):
    TARGET_POSITIONS = auto()
    TARGET_VALUES = auto()
    AD_PLACEMENT = auto()
    SWARM_ASSIGNMENT = auto()
    INTERCEPT_ASSIGNMENT = auto()
    AD_RESOLUTION = auto()
    TERMINAL = auto()


def _decode_ad_position(action: int) -> Tuple[int, int]:
    idx = action - AD_ACTION_BASE
    if not (0 <= idx < len(AD_POSITION_CANDIDATES)):
        raise ValueError("AD action outside candidate set")
    return AD_POSITION_CANDIDATES[idx]


def _decode_target_position_action(action: int) -> Tuple[int, int]:
    idx = action - TARGET_POSITION_ACTION_BASE
    if not (0 <= idx < len(TARGET_CANDIDATE_CELLS)):
        raise ValueError("Target action outside candidate set")
    return TARGET_CANDIDATE_CELLS[idx]


def _decode_target_value_action(action: int) -> Sequence[int]:
    idx = action - TARGET_VALUE_ACTION_BASE
    if not (0 <= idx < len(TARGET_VALUE_PERMUTATIONS)):
        raise ValueError("Value permutation index out of bounds")
    return TARGET_VALUE_PERMUTATIONS[idx]


def _decode_drone_action(action: int) -> Tuple[int, int, int, int]:
    rel = action - DRONE_ACTION_BASE
    per_entry = DRONE_TARGET_SLOTS * len(TOT_CHOICES)
    entry_index = rel // per_entry
    entry_col = entry_index
    if not (0 <= entry_col < GRID_SIZE):
        raise ValueError("Drone entry column out of bounds")
    target_index = (rel % per_entry) // len(TOT_CHOICES)
    tot_index = rel % len(TOT_CHOICES)
    return 0, entry_col, target_index, tot_index


def _decode_interceptor_action(action: int) -> Optional[int]:
    rel = action - INTERCEPT_ACTION_BASE
    if rel == NUM_ATTACKING_DRONES:
        return None
    return rel


def _arrival_time_to_point(plan: DronePlan, point: Tuple[int, int]) -> float:
    travel = math.dist((plan.entry_row, plan.entry_col), point)
    return TOT_CHOICES[plan.tot_idx] + travel


def _path_intersects(
    ad_pos: Tuple[int, int],
    entry: Tuple[int, int],
    target: Tuple[int, int],
) -> bool:
    steps = max(abs(entry[0] - target[0]), abs(entry[1] - target[1])) + 1
    if steps <= 0:
        steps = 1
    for step in range(steps + 1):
        t = step / steps
        row = entry[0] + (target[0] - entry[0]) * t
        col = entry[1] + (target[1] - entry[1]) * t
        if math.dist((row, col), ad_pos) <= AD_COVERAGE_RADIUS:
            return True
    return False


def _path_exposure_stats(
    ad_pos: Tuple[int, int],
    entry: Tuple[int, int],
    target: Tuple[int, int],
    samples: int = 80,
) -> Tuple[float, Optional[Tuple[float, float]]]:
    if not _path_intersects(ad_pos, entry, target):
        return 0.0, None
    exposure = 0.0
    entry_point: Optional[Tuple[float, float]] = None
    steps = max(samples, 10)
    prev_point = (float(entry[0]), float(entry[1]))
    prev_inside = math.dist(prev_point, ad_pos) <= AD_COVERAGE_RADIUS
    if prev_inside:
        entry_point = prev_point
    for step in range(1, steps + 1):
        t = step / steps
        point = (
            entry[0] + (target[0] - entry[0]) * t,
            entry[1] + (target[1] - entry[1]) * t,
        )
        inside = math.dist(point, ad_pos) <= AD_COVERAGE_RADIUS
        segment = math.dist(point, prev_point)
        if inside or prev_inside:
            exposure += segment
            if entry_point is None and inside:
                entry_point = point
        prev_point = point
        prev_inside = inside
    if entry_point is None:
        entry_point = point
    return exposure, entry_point


class SwarmDefenseState(pyspiel.State):
    def __init__(self, game: "SwarmDefenseGame"):
        super().__init__(game)
        self._phase = Phase.TARGET_POSITIONS
        self._history: List[int] = []
        self._target_positions: List[Tuple[int, int]] = []
        self._targets: List[TargetCluster] = []
        self._ad_units: List[ADUnit] = []
        self._drone_plans: List[DronePlan] = []
        self._interceptor_steps = 0
        self._pending_ad_targets: List[ADIntercept] = []
        self._next_ad_resolution_index = 0
        self._returns = [0.0, 0.0]

    def phase(self) -> Phase:
        return self._phase

    def snapshot(self) -> Dict[str, object]:
        return {
            "phase": self._phase.name,
            "targets": tuple(self._targets),
            "ad_units": tuple(
                {
                    "position": (unit.row, unit.col),
                    "alive": unit.alive,
                    "destroyed_by": unit.destroyed_by,
                }
                for unit in self._ad_units
            ),
            "drones": tuple(
                {
                    "entry": (plan.entry_row, plan.entry_col),
                    "target_idx": plan.target_idx,
                    "target_type": self._describe_target_type(plan.target_idx),
                    "destination": self._drone_destination(plan),
                    "target_value": (
                        self._targets[plan.target_idx].value
                        if plan.target_idx < len(self._targets)
                        else None
                    ),
                    "tot_idx": plan.tot_idx,
                    "tot": TOT_CHOICES[plan.tot_idx],
                    "destroyed_by": plan.destroyed_by,
                    "intercepts": tuple(plan.intercepts),
                }
                for plan in self._drone_plans
            ),
            "returns": tuple(self._returns),
        }

    def current_player(self) -> int:
        if self._phase == Phase.TERMINAL:
            return pyspiel.PlayerId.TERMINAL
        if self._phase in (Phase.TARGET_POSITIONS, Phase.TARGET_VALUES, Phase.AD_RESOLUTION):
            return pyspiel.PlayerId.CHANCE
        if self._phase in (Phase.AD_PLACEMENT, Phase.INTERCEPT_ASSIGNMENT):
            return 1
        return 0

    def legal_actions(self, player: Optional[int] = None) -> List[int]:
        if self._phase == Phase.TERMINAL:
            return []
        if self._phase == Phase.TARGET_POSITIONS:
            remaining = [
                idx
                for idx, cell in enumerate(TARGET_CANDIDATE_CELLS)
                if cell not in self._target_positions
            ]
            return [TARGET_POSITION_ACTION_BASE + idx for idx in remaining]
        if self._phase == Phase.TARGET_VALUES:
            return [
                TARGET_VALUE_ACTION_BASE + idx
                for idx in range(len(TARGET_VALUE_PERMUTATIONS))
            ]
        if self._phase == Phase.AD_PLACEMENT:
            occupied = {(unit.row, unit.col) for unit in self._ad_units}
            actions = []
            for idx, pos in enumerate(AD_POSITION_CANDIDATES):
                if pos not in occupied:
                    actions.append(AD_ACTION_BASE + idx)
            return actions
        if self._phase == Phase.SWARM_ASSIGNMENT:
            return [DRONE_ACTION_BASE + i for i in range(DRONE_ASSIGNMENT_ACTIONS)]
        if self._phase == Phase.INTERCEPT_ASSIGNMENT:
            choices = [
                INTERCEPT_ACTION_BASE + idx
                for idx, plan in enumerate(self._drone_plans)
                if plan.destroyed_by is None
            ]
            choices.append(INTERCEPT_ACTION_BASE + NUM_ATTACKING_DRONES)
            return choices
        if self._phase == Phase.AD_RESOLUTION:
            return [AD_RESOLVE_ACTION_BASE, AD_RESOLVE_ACTION_BASE + 1]
        return []

    def chance_outcomes(self) -> List[Tuple[int, float]]:
        if self._phase == Phase.TARGET_POSITIONS:
            remaining = [
                idx
                for idx, cell in enumerate(TARGET_CANDIDATE_CELLS)
                if cell not in self._target_positions
            ]
            probability = 1.0 / len(remaining)
            return [
                (TARGET_POSITION_ACTION_BASE + idx, probability)
                for idx in remaining
            ]
        if self._phase == Phase.TARGET_VALUES:
            probability = 1.0 / len(TARGET_VALUE_PERMUTATIONS)
            return [
                (TARGET_VALUE_ACTION_BASE + idx, probability)
                for idx in range(len(TARGET_VALUE_PERMUTATIONS))
            ]
        if self._phase == Phase.AD_RESOLUTION:
            intercept = self._pending_ad_targets[self._next_ad_resolution_index]
            hit_prob = intercept.probability
            miss_prob = max(0.0, 1.0 - hit_prob)
            return [
                (AD_RESOLVE_ACTION_BASE, miss_prob),
                (AD_RESOLVE_ACTION_BASE + 1, hit_prob),
            ]
        return []

    def apply_action(self, action: int) -> None:
        self._history.append(action)
        if self._phase == Phase.TARGET_POSITIONS:
            self._apply_target_position_action(action)
        elif self._phase == Phase.TARGET_VALUES:
            self._apply_target_value_action(action)
        elif self._phase == Phase.AD_PLACEMENT:
            self._apply_ad_action(action)
        elif self._phase == Phase.SWARM_ASSIGNMENT:
            self._apply_drone_action(action)
        elif self._phase == Phase.INTERCEPT_ASSIGNMENT:
            self._apply_interceptor_action(action)
        elif self._phase == Phase.AD_RESOLUTION:
            self._apply_ad_resolution(action)
        else:
            raise ValueError("Cannot apply actions in terminal state")

    def _apply_target_position_action(self, action: int) -> None:
        cell = _decode_target_position_action(action)
        if cell in self._target_positions:
            raise ValueError("Target cell already selected")
        self._target_positions.append(cell)
        if len(self._target_positions) == NUM_TARGETS:
            self._phase = Phase.TARGET_VALUES

    def _apply_target_value_action(self, action: int) -> None:
        perm = _decode_target_value_action(action)
        if len(self._target_positions) != NUM_TARGETS:
            raise ValueError("Target positions incomplete")
        self._targets = [
            TargetCluster(
                row=self._target_positions[i][0],
                col=self._target_positions[i][1],
                value=TARGET_VALUE_OPTIONS[perm[i]],
            )
            for i in range(NUM_TARGETS)
        ]
        self._phase = Phase.AD_PLACEMENT

    def _apply_ad_action(self, action: int) -> None:
        row, col = _decode_ad_position(action)
        if (row, col) in {(unit.row, unit.col) for unit in self._ad_units}:
            raise ValueError("AD position already occupied")
        self._ad_units.append(ADUnit(row=row, col=col))
        if len(self._ad_units) == NUM_AD_UNITS:
            self._phase = Phase.SWARM_ASSIGNMENT

    def _apply_drone_action(self, action: int) -> None:
        if len(self._drone_plans) >= NUM_ATTACKING_DRONES:
            raise ValueError("All drones already assigned")
        entry_row, entry_col, target_idx, tot_idx = _decode_drone_action(action)
        max_target_index = len(self._targets) + len(self._ad_units)
        if not (0 <= target_idx < max_target_index):
            raise ValueError("Invalid target index for drone plan")
        plan = DronePlan(entry_row, entry_col, target_idx, tot_idx)
        self._drone_plans.append(plan)
        if len(self._drone_plans) == NUM_ATTACKING_DRONES:
            self._phase = Phase.INTERCEPT_ASSIGNMENT

    def _apply_interceptor_action(self, action: int) -> None:
        if self._interceptor_steps >= NUM_INTERCEPTORS:
            raise ValueError("No interceptors remaining")
        drone_idx = _decode_interceptor_action(action)
        if drone_idx is not None:
            if not (0 <= drone_idx < len(self._drone_plans)):
                raise ValueError("Invalid drone index for interception")
            plan = self._drone_plans[drone_idx]
            if plan.destroyed_by is None:
                plan.destroyed_by = "interceptor"
        self._interceptor_steps += 1
        if self._interceptor_steps == NUM_INTERCEPTORS:
            self._start_ad_resolution()

    def _start_ad_resolution(self) -> None:
        self._process_ad_targeting_effects()
        self._pending_ad_targets.clear()
        engaged: set[int] = set()
        for ad_idx, ad_unit in enumerate(self._ad_units):
            if not ad_unit.alive:
                continue
            intercept = self._select_drone_for_ad(ad_idx, engaged)
            if intercept is not None:
                engaged.add(intercept.drone_idx)
                self._pending_ad_targets.append(intercept)
        if not self._pending_ad_targets:
            self._phase = Phase.TERMINAL
            self._finalize_returns()
        else:
            self._phase = Phase.AD_RESOLUTION
            self._next_ad_resolution_index = 0

    def _process_ad_targeting_effects(self) -> None:
        for ad_idx, ad_unit in enumerate(self._ad_units):
            if not ad_unit.alive:
                continue
            candidate = self._earliest_drone_targeting_ad(ad_idx)
            if candidate is None:
                continue
            ad_unit.alive = False
            ad_unit.destroyed_by = f"drone:{candidate}"
            plan = self._drone_plans[candidate]
            if plan.destroyed_by is None:
                plan.destroyed_by = f"ad_target:{ad_idx}"

    def _earliest_drone_targeting_ad(self, ad_idx: int) -> Optional[int]:
        best: Optional[Tuple[float, int]] = None
        ad_unit = self._ad_units[ad_idx]
        destination = (ad_unit.row, ad_unit.col)
        ad_target_index = len(self._targets) + ad_idx
        for idx, plan in enumerate(self._drone_plans):
            if plan.destroyed_by is not None:
                continue
            if plan.target_idx != ad_target_index:
                continue
            arrival = _arrival_time_to_point(plan, destination)
            if best is None or arrival < best[0] or (arrival == best[0] and idx < best[1]):
                best = (arrival, idx)
        return None if best is None else best[1]

    def _select_drone_for_ad(self, ad_idx: int, engaged: set[int]) -> Optional[ADIntercept]:
        ad_unit = self._ad_units[ad_idx]
        position = (ad_unit.row, ad_unit.col)
        best_sort: Optional[Tuple[float, float, int]] = None
        best_payload: Optional[Tuple[int, float, float, Tuple[float, float]]] = None
        for idx, plan in enumerate(self._drone_plans):
            if plan.destroyed_by is not None or idx in engaged:
                continue
            destination = self._drone_destination(plan)
            exposure, entry_point = _path_exposure_stats(
                position,
                (plan.entry_row, plan.entry_col),
                destination,
            )
            if exposure <= 0.0 or entry_point is None:
                continue
            probability = 1.0 - math.exp(-AD_KILL_RATE * exposure)
            probability = max(0.0, min(1.0, probability))
            if probability <= 0.0:
                continue
            arrival = _arrival_time_to_point(plan, position)
            sort_key = (arrival, -exposure, idx)
            if best_sort is None or sort_key < best_sort:
                best_sort = sort_key
                best_payload = (idx, exposure, probability, entry_point)
        if best_payload is None:
            return None
        drone_idx, exposure, probability, entry_point = best_payload
        return ADIntercept(
            ad_idx=ad_idx,
            drone_idx=drone_idx,
            exposure=exposure,
            probability=probability,
            hit_point=entry_point,
        )

    def _drone_destination(self, plan: DronePlan) -> Tuple[int, int]:
        if plan.target_idx < len(self._targets):
            target = self._targets[plan.target_idx]
            return (target.row, target.col)
        ad_idx = plan.target_idx - len(self._targets)
        if not (0 <= ad_idx < len(self._ad_units)):
            raise ValueError("Drone references unknown AD unit")
        ad_unit = self._ad_units[ad_idx]
        return (ad_unit.row, ad_unit.col)

    def _apply_ad_resolution(self, action: int) -> None:
        if not self._pending_ad_targets:
            raise ValueError("No AD engagements pending")
        intercept = self._pending_ad_targets[self._next_ad_resolution_index]
        plan = self._drone_plans[intercept.drone_idx]
        ad_unit = self._ad_units[intercept.ad_idx]
        success = action == AD_RESOLVE_ACTION_BASE + 1
        if success and plan.destroyed_by is None:
            plan.destroyed_by = f"ad:{intercept.ad_idx}"
            plan.intercepts.append((intercept.ad_idx, intercept.hit_point))
            ad_unit.intercept_log.append((intercept.drone_idx, intercept.hit_point))
        self._next_ad_resolution_index += 1
        if self._next_ad_resolution_index >= len(self._pending_ad_targets):
            self._phase = Phase.TERMINAL
            self._finalize_returns()

    def _finalize_returns(self) -> None:
        total_damage = 0.0
        for plan in self._drone_plans:
            if plan.destroyed_by is None and plan.target_idx < len(self._targets):
                total_damage += self._targets[plan.target_idx].value
        self._returns = [total_damage, -total_damage]

    def is_terminal(self) -> bool:
        return self._phase == Phase.TERMINAL

    def returns(self) -> List[float]:
        return list(self._returns)

    def observation_string(self, player: int) -> str:
        lines = [f"Phase: {self._phase.name}"]
        lines.append("Targets:")
        for idx, target in enumerate(self._targets):
            lines.append(f"  T{idx}: ({target.row},{target.col}) value={target.value}")
        lines.append("AD units:")
        for idx, unit in enumerate(self._ad_units):
            status = "alive" if unit.alive else f"destroyed({unit.destroyed_by})"
            lines.append(f"  AD{idx}: ({unit.row},{unit.col}) {status}")
        lines.append("Drone assignments:")
        for idx, plan in enumerate(self._drone_plans):
            dest = self._describe_target_type(plan.target_idx)
            lines.append(
                f"  D{idx}: entry=(0,{plan.entry_col}) -> {dest}"
                f" ToT={TOT_CHOICES[plan.tot_idx]} destroyed_by={plan.destroyed_by}"
            )
        lines.append(f"Returns: {self._returns}")
        return "\n".join(lines)

    def information_state_string(self, player: int) -> str:
        return self.observation_string(player)

    def action_to_string(self, player: Optional[int], action: int) -> str:
        if TARGET_POSITION_ACTION_BASE <= action < TARGET_VALUE_ACTION_BASE:
            row, col = _decode_target_position_action(action)
            return f"target_cell:({row},{col})"
        if TARGET_VALUE_ACTION_BASE <= action < AD_ACTION_BASE:
            perm = _decode_target_value_action(action)
            return "target_values:" + ",".join(str(TARGET_VALUE_OPTIONS[idx]) for idx in perm)
        if AD_ACTION_BASE <= action < DRONE_ACTION_BASE:
            row, col = _decode_ad_position(action)
            return f"ad_place:({row},{col})"
        if DRONE_ACTION_BASE <= action < INTERCEPT_ACTION_BASE:
            entry_row, entry_col, target_idx, tot_idx = _decode_drone_action(action)
            target_desc = self._describe_target_type(target_idx)
            return (
                f"drone_assign:entry=({entry_row},{entry_col}) target={target_desc}"
                f" ToT={TOT_CHOICES[tot_idx]}"
            )
        if INTERCEPT_ACTION_BASE <= action < AD_RESOLVE_ACTION_BASE:
            choice = _decode_interceptor_action(action)
            if choice is None:
                return "interceptor:pass"
            return f"interceptor:drone={choice}"
        if AD_RESOLVE_ACTION_BASE <= action < NUM_DISTINCT_ACTIONS:
            return "ad_resolution:success" if action == AD_RESOLVE_ACTION_BASE + 1 else "ad_resolution:fail"
        return f"unknown_action:{action}"

    def _describe_target_type(self, target_idx: int) -> str:
        if target_idx < len(self._targets):
            return f"target:{target_idx}"
        ad_idx = target_idx - len(self._targets)
        return f"ad:{ad_idx}"

    def __str__(self) -> str:
        return self.observation_string(0)


_GAME_TYPE = pyspiel.GameType(
    short_name="swarm_defense",
    long_name="Swarm Defense Sequential Game",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=2,
    min_num_players=2,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=False,
    parameter_specification={},
)

_MAX_DAMAGE = NUM_ATTACKING_DRONES * max(TARGET_VALUE_OPTIONS)

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=NUM_DISTINCT_ACTIONS,
    max_chance_outcomes=max(len(TARGET_CANDIDATE_CELLS), len(TARGET_VALUE_PERMUTATIONS), 2),
    num_players=2,
    min_utility=-float(_MAX_DAMAGE),
    max_utility=float(_MAX_DAMAGE),
    utility_sum=0.0,
    max_game_length=
    NUM_TARGETS  # target positions
    + 1  # value permutation
    + NUM_AD_UNITS
    + NUM_ATTACKING_DRONES
    + NUM_INTERCEPTORS
    + NUM_AD_UNITS,
)


class SwarmDefenseGame(pyspiel.Game):
    def __init__(self, params: Optional[Dict[str, int]] = None):
        super().__init__(_GAME_TYPE, _GAME_INFO, params or {})

    def new_initial_state(self) -> SwarmDefenseState:
        return SwarmDefenseState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        return pyspiel.IIGObserverForPublicInfo(iig_obs_type, params)


pyspiel.register_game(_GAME_TYPE, SwarmDefenseGame)

__all__ = [
    "SwarmDefenseGame",
    "SwarmDefenseState",
    "TargetCluster",
    "ADUnit",
    "DronePlan",
    "Phase",
    "TOT_CHOICES",
]
