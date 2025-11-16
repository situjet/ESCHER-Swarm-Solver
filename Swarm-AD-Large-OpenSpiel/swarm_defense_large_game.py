"""Non-abstracted Swarm Defense OpenSpiel game with richer dynamics."""
from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pyspiel

try:
    from .blueprint_midpoints import (
        MIDPOINT_STRATEGIES,
        BlueprintContext,
        blueprint_midpoints,
    )
    from .pathfinding import (
        Bounds,
        CircleObstacle,
        path_length,
        rrt_path,
        sample_path,
        smooth_path,
    )
except ImportError:  # pragma: no cover - fallback for script execution
    from blueprint_midpoints import (  # type: ignore
        MIDPOINT_STRATEGIES,
        BlueprintContext,
        blueprint_midpoints,
    )
    from pathfinding import (  # type: ignore
        Bounds,
        CircleObstacle,
        path_length,
        rrt_path,
        sample_path,
        smooth_path,
    )

ARENA_WIDTH = 32.0
ARENA_HEIGHT = 32.0
BOTTOM_START = ARENA_HEIGHT * 0.5
NUM_TARGETS = 3
NUM_AD_UNITS = 2
NUM_ATTACKING_DRONES = 40
NUM_INTERCEPTORS = NUM_ATTACKING_DRONES // 2
DRONE_SPEED = 1.0
INTERCEPTOR_SPEED_MULTIPLIER = 2.0
ENTRY_LANES = 24
ENTRY_POINTS: List[Tuple[float, float]] = [(0.0, lane + 0.5) for lane in range(ENTRY_LANES)]
LARGE_TOT_CHOICES: Tuple[float, ...] = (0.0, 1.5, 3.0, 4.5)
AD_COVERAGE_RADIUS = 5.5
AD_KILL_RATE = .05
AD_MIN_EFFECTIVE_EXPOSURE = 0.75
INTERCEPTOR_LAUNCH_ROW = ARENA_HEIGHT + 2.0
INTERCEPTOR_KILL_PROB = 0.95
DRONE_VS_AD_KILL_PROB = 0.8
DRONE_VS_TARGET_KILL_PROB = 0.7
RRT_TIME_LIMIT_SEC = 0.035

TARGET_CANDIDATE_CELLS: List[Tuple[float, float]] = [
    (row, col)
    for row in [12.0, 14.5, 17.0, 19.5, 22.0]
    for col in [3.0, 7.0, 11.0, 15.0, 19.0]
]
TARGET_VALUE_OPTIONS: Tuple[float, ...] = (10.0, 25.0, 60.0)
TARGET_VALUE_PERMUTATIONS: List[Sequence[int]] = list(
    itertools.permutations(range(NUM_TARGETS))
)

AD_POSITION_CANDIDATES: List[Tuple[float, float]] = [
    (row, col)
    for row in [13.0, 15.5, 18.0, 20.5]
    for col in [2.5, 6.5, 10.5, 14.5, 18.5]
]

DRONE_TARGET_SLOTS = NUM_TARGETS + NUM_AD_UNITS
MIDPOINT_OPTIONS = len(MIDPOINT_STRATEGIES)

TARGET_POSITION_ACTIONS = len(TARGET_CANDIDATE_CELLS)
TARGET_VALUE_ACTIONS = len(TARGET_VALUE_PERMUTATIONS)
AD_PLACEMENT_ACTIONS = len(AD_POSITION_CANDIDATES)
DRONE_ASSIGNMENT_ACTIONS = (
    ENTRY_LANES * DRONE_TARGET_SLOTS * len(LARGE_TOT_CHOICES) * MIDPOINT_OPTIONS
)
INTERCEPT_CHOICES = NUM_ATTACKING_DRONES + 1

TARGET_POSITION_ACTION_BASE = 0
TARGET_VALUE_ACTION_BASE = TARGET_POSITION_ACTION_BASE + TARGET_POSITION_ACTIONS
AD_ACTION_BASE = TARGET_VALUE_ACTION_BASE + TARGET_VALUE_ACTIONS
DRONE_ACTION_BASE = AD_ACTION_BASE + AD_PLACEMENT_ACTIONS
INTERCEPT_ACTION_BASE = DRONE_ACTION_BASE + DRONE_ASSIGNMENT_ACTIONS
AD_RESOLVE_ACTION_BASE = INTERCEPT_ACTION_BASE + INTERCEPT_CHOICES
NUM_DISTINCT_ACTIONS = AD_RESOLVE_ACTION_BASE + 2

BOUNDS = Bounds(0.0, ARENA_HEIGHT, 0.0, ARENA_WIDTH)


def _degrees_to_radians(value: float) -> float:
    return math.radians(value)


def _normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _angle_between(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.atan2(b[0] - a[0], b[1] - a[1])


@dataclass(frozen=True)
class TargetCluster:
    row: float
    col: float
    value: float


@dataclass
class ADUnit:
    row: float
    col: float
    alive: bool = True
    destroyed_by: Optional[str] = None
    orientation: float = math.pi / 2
    intercept_log: List[Tuple[int, Tuple[float, float], float]] = field(default_factory=list)


@dataclass
class DronePlan:
    entry_lane_idx: int
    target_idx: int
    tot_idx: int
    blueprint_idx: int
    destroyed_by: Optional[str] = None
    intercepts: List[Tuple[int, Tuple[float, float], float]] = field(default_factory=list)
    interceptor_hit: Optional[Tuple[float, float]] = None
    interceptor_time: Optional[float] = None
    waypoints: List[Tuple[float, float]] = field(default_factory=list)
    path_samples: List[Tuple[float, float, float]] = field(default_factory=list)
    total_distance: float = 0.0
    hold_time: float = 0.0
    arrival_time: float = 0.0
    strike_success: Optional[bool] = None
    damage_inflicted: float = 0.0

    @property
    def entry_point(self) -> Tuple[float, float]:
        return ENTRY_POINTS[self.entry_lane_idx]


@dataclass
class ADIntercept:
    ad_idx: int
    drone_idx: int
    exposure: float
    probability: float
    hit_point: Tuple[float, float]
    intercept_time: float


@dataclass
class InterceptorEngagement:
    drone_idx: int
    hit_point: Tuple[float, float]
    intercept_time: float
    probability: float = INTERCEPTOR_KILL_PROB


@dataclass
class DroneADStrike:
    ad_idx: int
    drone_idx: int
    probability: float = DRONE_VS_AD_KILL_PROB


@dataclass
class DroneTargetStrike:
    drone_idx: int
    target_idx: int
    probability: float = DRONE_VS_TARGET_KILL_PROB


class Phase(Enum):
    TARGET_POSITIONS = auto()
    TARGET_VALUES = auto()
    AD_PLACEMENT = auto()
    SWARM_ASSIGNMENT = auto()
    INTERCEPT_ASSIGNMENT = auto()
    INTERCEPT_RESOLUTION = auto()
    AD_RESOLUTION = auto()
    DRONE_AD_STRIKE_RESOLUTION = auto()
    TARGET_DAMAGE_RESOLUTION = auto()
    TERMINAL = auto()


def decode_target_position_action(action: int) -> Tuple[float, float]:
    idx = action - TARGET_POSITION_ACTION_BASE
    if not (0 <= idx < TARGET_POSITION_ACTIONS):
        raise ValueError("Target position action out of range")
    return TARGET_CANDIDATE_CELLS[idx]


def decode_target_value_action(action: int) -> Sequence[int]:
    idx = action - TARGET_VALUE_ACTION_BASE
    if not (0 <= idx < TARGET_VALUE_ACTIONS):
        raise ValueError("Target value action out of range")
    return TARGET_VALUE_PERMUTATIONS[idx]


def decode_ad_position_action(action: int) -> Tuple[float, float]:
    idx = action - AD_ACTION_BASE
    if not (0 <= idx < AD_PLACEMENT_ACTIONS):
        raise ValueError("AD placement action out of range")
    return AD_POSITION_CANDIDATES[idx]


def decode_drone_action(action: int) -> Tuple[int, int, int, int]:
    rel = action - DRONE_ACTION_BASE
    if not (0 <= rel < DRONE_ASSIGNMENT_ACTIONS):
        raise ValueError("Drone action out of range")
    per_entry = DRONE_TARGET_SLOTS * len(LARGE_TOT_CHOICES) * MIDPOINT_OPTIONS
    entry_idx = rel // per_entry
    rem = rel % per_entry
    per_target = len(LARGE_TOT_CHOICES) * MIDPOINT_OPTIONS
    target_idx = rem // per_target
    rem = rem % per_target
    tot_idx = rem // MIDPOINT_OPTIONS
    blueprint_idx = rem % MIDPOINT_OPTIONS
    return entry_idx, target_idx, tot_idx, blueprint_idx


def encode_drone_action(entry_idx: int, target_idx: int, tot_idx: int, blueprint_idx: int) -> int:
    per_entry = DRONE_TARGET_SLOTS * len(LARGE_TOT_CHOICES) * MIDPOINT_OPTIONS
    per_target = len(LARGE_TOT_CHOICES) * MIDPOINT_OPTIONS
    rel = entry_idx * per_entry + target_idx * per_target + tot_idx * MIDPOINT_OPTIONS + blueprint_idx
    return DRONE_ACTION_BASE + rel


def decode_interceptor_action(action: int) -> Optional[int]:
    rel = action - INTERCEPT_ACTION_BASE
    if not (0 <= rel < INTERCEPT_CHOICES):
        raise ValueError("Interceptor action out of range")
    if rel == NUM_ATTACKING_DRONES:
        return None
    return rel


class SwarmDefenseLargeState(pyspiel.State):
    def __init__(self, game: "SwarmDefenseLargeGame"):
        super().__init__(game)
        self._phase = Phase.TARGET_POSITIONS
        self._history: List[int] = []
        self._targets: List[TargetCluster] = []
        self._target_destroyed: List[bool] = []
        self._target_positions: List[Tuple[float, float]] = []
        self._ad_units: List[ADUnit] = []
        self._drone_plans: List[DronePlan] = []
        self._interceptor_steps = 0
        self._pending_interceptor_hits: List[InterceptorEngagement] = []
        self._next_interceptor_resolution_index = 0
        self._interceptor_engaged: set[int] = set()
        self._pending_ad_targets: List[ADIntercept] = []
        self._next_ad_resolution_index = 0
        self._pending_ad_strikes: List[DroneADStrike] = []
        self._next_ad_strike_index = 0
        self._pending_target_strikes: List[DroneTargetStrike] = []
        self._next_target_strike_index = 0
        self._damage_from_targets = 0.0
        self._returns = [0.0, 0.0]
        self._rng = random.Random()
        self._ad_orientation_events: Dict[int, List[Tuple[float, float]]] = {}
        self._tot_anchor = None

    def phase(self) -> Phase:
        return self._phase

    def snapshot(self) -> Dict[str, object]:
        return {
            "phase": self._phase.name,
            "targets": tuple(self._targets),
            "target_destroyed": tuple(self._target_destroyed),
            "ad_units": tuple(
                {
                    "position": (unit.row, unit.col),
                    "alive": unit.alive,
                    "destroyed_by": unit.destroyed_by,
                    "orientation": unit.orientation,
                    "intercept_log": tuple(unit.intercept_log),
                    "orientation_events": tuple(self._ad_orientation_events.get(idx, [])),
                }
                for idx, unit in enumerate(self._ad_units)
            ),
            "drones": tuple(
                {
                    "entry": plan.entry_point,
                    "target_idx": plan.target_idx,
                    "destination": self._drone_destination(plan),
                    "target_value": (
                        self._targets[plan.target_idx].value
                        if plan.target_idx < len(self._targets)
                        else None
                    ),
                    "tot_idx": plan.tot_idx,
                    "tot": LARGE_TOT_CHOICES[plan.tot_idx],
                    "hold_time": plan.hold_time,
                    "arrival_time": plan.arrival_time,
                    "blueprint_idx": plan.blueprint_idx,
                    "blueprint": MIDPOINT_STRATEGIES[plan.blueprint_idx],
                    "path": tuple(plan.waypoints),
                    "path_samples": tuple(plan.path_samples),
                    "destroyed_by": plan.destroyed_by,
                    "intercepts": tuple(plan.intercepts),
                    "interceptor_hit": plan.interceptor_hit,
                    "interceptor_time": plan.interceptor_time,
                    "strike_success": plan.strike_success,
                    "damage_inflicted": plan.damage_inflicted,
                    "total_distance": plan.total_distance,
                }
                for plan in self._drone_plans
            ),
            "returns": tuple(self._returns),
        }

    def current_player(self) -> int:
        if self._phase == Phase.TERMINAL:
            return pyspiel.PlayerId.TERMINAL
        if self._phase in (
            Phase.TARGET_POSITIONS,
            Phase.TARGET_VALUES,
            Phase.INTERCEPT_RESOLUTION,
            Phase.AD_RESOLUTION,
            Phase.DRONE_AD_STRIKE_RESOLUTION,
            Phase.TARGET_DAMAGE_RESOLUTION,
        ):
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
            return [TARGET_VALUE_ACTION_BASE + idx for idx in range(TARGET_VALUE_ACTIONS)]
        if self._phase == Phase.AD_PLACEMENT:
            occupied = {(unit.row, unit.col) for unit in self._ad_units}
            actions = []
            for idx, pos in enumerate(AD_POSITION_CANDIDATES):
                if pos not in occupied:
                    actions.append(AD_ACTION_BASE + idx)
            return actions
        if self._phase == Phase.SWARM_ASSIGNMENT:
            if len(self._drone_plans) >= NUM_ATTACKING_DRONES:
                return []
            return [DRONE_ACTION_BASE + i for i in range(DRONE_ASSIGNMENT_ACTIONS)]
        if self._phase == Phase.INTERCEPT_ASSIGNMENT:
            choices = [
                INTERCEPT_ACTION_BASE + idx
                for idx, plan in enumerate(self._drone_plans)
                if plan.destroyed_by is None and idx not in self._interceptor_engaged
            ]
            choices.append(INTERCEPT_ACTION_BASE + NUM_ATTACKING_DRONES)
            return choices
        if self._phase in (
            Phase.INTERCEPT_RESOLUTION,
            Phase.AD_RESOLUTION,
            Phase.DRONE_AD_STRIKE_RESOLUTION,
            Phase.TARGET_DAMAGE_RESOLUTION,
        ):
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
            probability = 1.0 / TARGET_VALUE_ACTIONS
            return [
                (TARGET_VALUE_ACTION_BASE + idx, probability)
                for idx in range(TARGET_VALUE_ACTIONS)
            ]
        if self._phase == Phase.INTERCEPT_RESOLUTION:
            engagement = self._pending_interceptor_hits[self._next_interceptor_resolution_index]
            hit_prob = engagement.probability
            miss_prob = max(0.0, 1.0 - hit_prob)
            return [
                (AD_RESOLVE_ACTION_BASE, miss_prob),
                (AD_RESOLVE_ACTION_BASE + 1, hit_prob),
            ]
        if self._phase == Phase.AD_RESOLUTION:
            intercept = self._pending_ad_targets[self._next_ad_resolution_index]
            hit_prob = intercept.probability
            miss_prob = max(0.0, 1.0 - hit_prob)
            return [
                (AD_RESOLVE_ACTION_BASE, miss_prob),
                (AD_RESOLVE_ACTION_BASE + 1, hit_prob),
            ]
        if self._phase == Phase.DRONE_AD_STRIKE_RESOLUTION:
            strike = self._pending_ad_strikes[self._next_ad_strike_index]
            success = strike.probability
            failure = max(0.0, 1.0 - success)
            return [
                (AD_RESOLVE_ACTION_BASE, failure),
                (AD_RESOLVE_ACTION_BASE + 1, success),
            ]
        if self._phase == Phase.TARGET_DAMAGE_RESOLUTION:
            strike = self._pending_target_strikes[self._next_target_strike_index]
            success = strike.probability
            failure = max(0.0, 1.0 - success)
            return [
                (AD_RESOLVE_ACTION_BASE, failure),
                (AD_RESOLVE_ACTION_BASE + 1, success),
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
        elif self._phase == Phase.INTERCEPT_RESOLUTION:
            self._apply_interceptor_resolution(action)
        elif self._phase == Phase.AD_RESOLUTION:
            self._apply_ad_resolution(action)
        elif self._phase == Phase.DRONE_AD_STRIKE_RESOLUTION:
            self._apply_drone_ad_strike(action)
        elif self._phase == Phase.TARGET_DAMAGE_RESOLUTION:
            self._apply_target_damage_resolution(action)
        else:
            raise ValueError("Cannot apply actions in terminal state")

    def _apply_target_position_action(self, action: int) -> None:
        cell = decode_target_position_action(action)
        if cell in self._target_positions:
            raise ValueError("Target already placed at cell")
        self._target_positions.append(cell)
        if len(self._target_positions) == NUM_TARGETS:
            self._phase = Phase.TARGET_VALUES

    def _apply_target_value_action(self, action: int) -> None:
        perm = decode_target_value_action(action)
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
        self._target_destroyed = [False] * len(self._targets)
        self._phase = Phase.AD_PLACEMENT

    def _apply_ad_action(self, action: int) -> None:
        pos = decode_ad_position_action(action)
        if pos in {(unit.row, unit.col) for unit in self._ad_units}:
            raise ValueError("AD slot occupied")
        self._ad_units.append(ADUnit(row=pos[0], col=pos[1]))
        if len(self._ad_units) == NUM_AD_UNITS:
            self._phase = Phase.SWARM_ASSIGNMENT

    def _apply_drone_action(self, action: int) -> None:
        if len(self._drone_plans) >= NUM_ATTACKING_DRONES:
            raise ValueError("All drones assigned")
        entry_idx, target_idx, tot_idx, blueprint_idx = decode_drone_action(action)
        max_target_index = len(self._targets) + len(self._ad_units)
        if not (0 <= target_idx < max_target_index):
            raise ValueError("Invalid target specified for drone")
        plan = DronePlan(entry_idx, target_idx, tot_idx, blueprint_idx)
        self._initialize_drone_path(plan)
        self._drone_plans.append(plan)
        if len(self._drone_plans) == NUM_ATTACKING_DRONES:
            self._finalize_swarm_assignment()

    def _finalize_swarm_assignment(self) -> None:
        self._synchronize_tot_schedule()
        self._phase = Phase.INTERCEPT_ASSIGNMENT

    def _synchronize_tot_schedule(self) -> None:
        if not self._drone_plans:
            self._tot_anchor = 0.0
            return
        requirements = [
            (plan.total_distance / DRONE_SPEED) - LARGE_TOT_CHOICES[plan.tot_idx]
            for plan in self._drone_plans
        ]
        anchor = max(0.0, *requirements)
        self._tot_anchor = anchor
        for plan in self._drone_plans:
            travel_time = plan.total_distance / DRONE_SPEED
            desired_arrival = anchor + LARGE_TOT_CHOICES[plan.tot_idx]
            hold_time = max(0.0, desired_arrival - travel_time)
            plan.hold_time = hold_time
            plan.arrival_time = hold_time + travel_time

    def _initialize_drone_path(self, plan: DronePlan) -> None:
        destination = self._drone_destination(plan)
        entry = plan.entry_point
        strategy = MIDPOINT_STRATEGIES[plan.blueprint_idx]
        ctx = BlueprintContext(
            entry=entry,
            destination=destination,
            tot_delay=LARGE_TOT_CHOICES[plan.tot_idx],
            arena_width=ARENA_WIDTH,
            arena_height=ARENA_HEIGHT,
            ad_positions=[(unit.row, unit.col) for unit in self._ad_units],
        )
        bias_points = [entry] + blueprint_midpoints(strategy, ctx, self._rng)
        bias_points.append(destination)
        obstacles = self._obstacles()
        raw = rrt_path(
            entry,
            destination,
            obstacles,
            BOUNDS,
            rng=self._rng,
            bias_points=bias_points,
            time_limit_sec=RRT_TIME_LIMIT_SEC,
        )
        refined = smooth_path(raw, obstacles, rng=self._rng)
        plan.waypoints = refined
        plan.path_samples = sample_path(refined)
        plan.total_distance = plan.path_samples[-1][2] if plan.path_samples else path_length(refined)

    def _obstacles(self) -> List[CircleObstacle]:
        obstacles = [
            CircleObstacle(center=(unit.row, unit.col), radius=AD_COVERAGE_RADIUS, padding=0.6)
            for unit in self._ad_units
        ]
        return obstacles

    def _apply_interceptor_action(self, action: int) -> None:
        if self._interceptor_steps >= NUM_INTERCEPTORS:
            raise ValueError("No interceptors remaining")
        drone_idx = decode_interceptor_action(action)
        if drone_idx is not None:
            if not (0 <= drone_idx < len(self._drone_plans)):
                raise ValueError("Invalid drone index")
            plan = self._drone_plans[drone_idx]
            if plan.destroyed_by is None and drone_idx not in self._interceptor_engaged:
                destination = self._drone_destination(plan)
                arrival_time = self._arrival_time(plan, plan.total_distance)
                intercept = self._compute_interceptor_intercept(plan, destination)
                if intercept is not None:
                    intercept_time, hit_point = intercept
                    if intercept_time < arrival_time:
                        self._pending_interceptor_hits.append(
                            InterceptorEngagement(
                                drone_idx=drone_idx,
                                hit_point=hit_point,
                                intercept_time=intercept_time,
                            )
                        )
                        self._interceptor_engaged.add(drone_idx)
        self._interceptor_steps += 1
        if self._interceptor_steps == NUM_INTERCEPTORS:
            self._start_post_interceptor_resolution()

    def _arrival_time(self, plan: DronePlan, distance: float) -> float:
        travel = distance / DRONE_SPEED
        return plan.hold_time + travel

    def _start_post_interceptor_resolution(self) -> None:
        if self._pending_interceptor_hits:
            self._phase = Phase.INTERCEPT_RESOLUTION
            self._next_interceptor_resolution_index = 0
        else:
            self._start_ad_resolution()

    def _start_ad_resolution(self) -> None:
        if self._tot_anchor is None:
            self._synchronize_tot_schedule()
        self._pending_ad_targets.clear()
        engaged: set[int] = set()
        engagements: List[ADIntercept] = []
        for ad_idx, ad_unit in enumerate(self._ad_units):
            if not ad_unit.alive:
                continue
            while True:
                intercept = self._select_drone_for_ad(ad_idx, engaged)
                if intercept is None:
                    break
                engaged.add(intercept.drone_idx)
                engagements.append(intercept)
        engagements.sort(key=lambda intercept: (intercept.intercept_time, intercept.ad_idx, intercept.drone_idx))
        self._pending_ad_targets.extend(engagements)
        for events in self._ad_orientation_events.values():
            events.sort(key=lambda entry: entry[0])
        if not self._pending_ad_targets:
            self._start_drone_ad_strike_resolution()
        else:
            self._phase = Phase.AD_RESOLUTION
            self._next_ad_resolution_index = 0

    def _apply_interceptor_resolution(self, action: int) -> None:
        if not self._pending_interceptor_hits:
            raise ValueError("No interceptor engagements pending")
        engagement = self._pending_interceptor_hits[self._next_interceptor_resolution_index]
        plan = self._drone_plans[engagement.drone_idx]
        success = action == AD_RESOLVE_ACTION_BASE + 1
        if success and plan.destroyed_by is None:
            plan.destroyed_by = "interceptor"
            plan.interceptor_hit = engagement.hit_point
            plan.interceptor_time = engagement.intercept_time
        self._next_interceptor_resolution_index += 1
        if self._next_interceptor_resolution_index >= len(self._pending_interceptor_hits):
            self._pending_interceptor_hits.clear()
            self._next_interceptor_resolution_index = 0
            self._interceptor_engaged.clear()
            self._start_ad_resolution()

    def _earliest_drone_targeting_ad(self, ad_idx: int) -> Optional[int]:
        best: Optional[Tuple[float, int]] = None
        target_index = len(self._targets) + ad_idx
        for idx, plan in enumerate(self._drone_plans):
            if plan.destroyed_by is not None:
                continue
            if plan.target_idx != target_index:
                continue
            arrival = self._arrival_time(plan, plan.total_distance)
            if best is None or arrival < best[0] or (arrival == best[0] and idx < best[1]):
                best = (arrival, idx)
        return None if best is None else best[1]

    def _select_drone_for_ad(self, ad_idx: int, engaged: set[int]) -> Optional[ADIntercept]:
        ad_unit = self._ad_units[ad_idx]
        position = (ad_unit.row, ad_unit.col)
        best_sort: Optional[Tuple[float, float, int]] = None
        best_payload: Optional[Tuple[int, Tuple[float, float], float, float, float]] = None
        for idx, plan in enumerate(self._drone_plans):
            if plan.destroyed_by is not None or idx in engaged:
                continue
            exposure, entry_point, entry_distance = self._path_exposure_stats(plan, position)
            if exposure <= 0.0 or entry_point is None or entry_distance is None:
                continue
            intercept_time = self._arrival_time(plan, entry_distance)
            arrival_time = self._arrival_time(plan, plan.total_distance)
            direction = _angle_between(position, entry_point)
            if intercept_time >= arrival_time:
                continue
            distance_to_entry = math.dist(position, entry_point)
            sort_key = (distance_to_entry, intercept_time, idx)
            if best_sort is None or sort_key < best_sort:
                best_sort = sort_key
                best_payload = (idx, entry_point, intercept_time, exposure, direction)
        if best_payload is None:
            return None
        drone_idx, entry_point, intercept_time, exposure, direction = best_payload
        effective = max(exposure, AD_MIN_EFFECTIVE_EXPOSURE)
        probability = 1.0 - math.exp(-AD_KILL_RATE * effective)
        probability = max(0.0, min(1.0, probability))
        intercept = ADIntercept(
            ad_idx=ad_idx,
            drone_idx=drone_idx,
            exposure=exposure,
            probability=probability,
            hit_point=entry_point,
            intercept_time=intercept_time,
        )
        self._ad_units[ad_idx].orientation = direction
        self._record_orientation_event(ad_idx, intercept_time, direction)
        return intercept

    def _path_exposure_stats(
        self, plan: DronePlan, ad_pos: Tuple[float, float]
    ) -> Tuple[float, Optional[Tuple[float, float]], Optional[float]]:
        if not plan.path_samples:
            return 0.0, None, None
        exposure = 0.0
        entry_point = None
        entry_distance = None
        prev_sample = plan.path_samples[0]
        prev_inside = math.dist((prev_sample[0], prev_sample[1]), ad_pos) <= AD_COVERAGE_RADIUS
        for sample in plan.path_samples[1:]:
            point = (sample[0], sample[1])
            inside = math.dist(point, ad_pos) <= AD_COVERAGE_RADIUS
            segment = sample[2] - prev_sample[2]
            if inside or prev_inside:
                exposure += segment
                if entry_point is None and inside:
                    entry_point = point
                    entry_distance = sample[2]
            prev_sample = sample
            prev_inside = inside
        return exposure, entry_point, entry_distance

    def _compute_interceptor_intercept(
        self, plan: DronePlan, destination: Tuple[float, float]
    ) -> Optional[Tuple[float, Tuple[float, float]]]:
        launch = (INTERCEPTOR_LAUNCH_ROW, plan.entry_point[1])
        for sample in plan.path_samples:
            point = (sample[0], sample[1])
            drone_time = self._arrival_time(plan, sample[2])
            interceptor_distance = math.dist(launch, point)
            interceptor_time = interceptor_distance / (DRONE_SPEED * INTERCEPTOR_SPEED_MULTIPLIER)
            if drone_time >= interceptor_time:
                return drone_time, point
        return None

    def _drone_destination(self, plan: DronePlan) -> Tuple[float, float]:
        if plan.target_idx < len(self._targets):
            target = self._targets[plan.target_idx]
            return (target.row, target.col)
        ad_idx = plan.target_idx - len(self._targets)
        if not (0 <= ad_idx < len(self._ad_units)):
            raise ValueError("Drone references unknown AD unit")
        ad_unit = self._ad_units[ad_idx]
        return (ad_unit.row, ad_unit.col)

    def _record_orientation_event(self, ad_idx: int, intercept_time: float, direction: float) -> None:
        events = self._ad_orientation_events.setdefault(ad_idx, [])
        events.append((intercept_time, direction))

    def _apply_ad_resolution(self, action: int) -> None:
        if not self._pending_ad_targets:
            raise ValueError("No engagements pending")
        intercept = self._pending_ad_targets[self._next_ad_resolution_index]
        plan = self._drone_plans[intercept.drone_idx]
        ad_unit = self._ad_units[intercept.ad_idx]
        success = action == AD_RESOLVE_ACTION_BASE + 1
        arrival_time = self._arrival_time(plan, plan.total_distance)
        if success and plan.destroyed_by is None and intercept.intercept_time < arrival_time:
            plan.destroyed_by = f"ad:{intercept.ad_idx}"
            plan.intercepts.append((intercept.ad_idx, intercept.hit_point, intercept.intercept_time))
            ad_unit.intercept_log.append((intercept.drone_idx, intercept.hit_point, intercept.intercept_time))
        self._next_ad_resolution_index += 1
        if self._next_ad_resolution_index >= len(self._pending_ad_targets):
            self._pending_ad_targets.clear()
            self._next_ad_resolution_index = 0
            self._start_drone_ad_strike_resolution()

    def _start_drone_ad_strike_resolution(self) -> None:
        self._pending_ad_strikes.clear()
        self._next_ad_strike_index = 0
        for ad_idx, ad_unit in enumerate(self._ad_units):
            if not ad_unit.alive:
                continue
            candidate = self._earliest_drone_targeting_ad(ad_idx)
            if candidate is None:
                continue
            plan = self._drone_plans[candidate]
            if plan.destroyed_by is not None:
                continue
            self._pending_ad_strikes.append(DroneADStrike(ad_idx=ad_idx, drone_idx=candidate))
        if self._pending_ad_strikes:
            self._phase = Phase.DRONE_AD_STRIKE_RESOLUTION
        else:
            self._start_target_damage_resolution()

    def _apply_drone_ad_strike(self, action: int) -> None:
        if not self._pending_ad_strikes:
            raise ValueError("No drone vs AD strikes pending")
        strike = self._pending_ad_strikes[self._next_ad_strike_index]
        ad_unit = self._ad_units[strike.ad_idx]
        plan = self._drone_plans[strike.drone_idx]
        success = action == AD_RESOLVE_ACTION_BASE + 1
        if plan.destroyed_by is None:
            plan.destroyed_by = f"ad_target:{strike.ad_idx}"
        plan.strike_success = success
        if success and ad_unit.alive:
            ad_unit.alive = False
            ad_unit.destroyed_by = f"drone:{strike.drone_idx}"
        self._next_ad_strike_index += 1
        if self._next_ad_strike_index >= len(self._pending_ad_strikes):
            self._pending_ad_strikes.clear()
            self._next_ad_strike_index = 0
            self._start_target_damage_resolution()

    def _start_target_damage_resolution(self) -> None:
        self._pending_target_strikes.clear()
        self._next_target_strike_index = 0
        self._damage_from_targets = 0.0
        if len(self._target_destroyed) != len(self._targets):
            self._target_destroyed = [False] * len(self._targets)
        for idx, plan in enumerate(self._drone_plans):
            if plan.target_idx < len(self._targets):
                plan.damage_inflicted = 0.0
                plan.strike_success = None
                if plan.destroyed_by is None:
                    self._pending_target_strikes.append(
                        DroneTargetStrike(drone_idx=idx, target_idx=plan.target_idx)
                    )
            else:
                plan.damage_inflicted = 0.0
        if self._pending_target_strikes:
            self._phase = Phase.TARGET_DAMAGE_RESOLUTION
        else:
            self._phase = Phase.TERMINAL
            self._finalize_returns()

    def _apply_target_damage_resolution(self, action: int) -> None:
        if not self._pending_target_strikes:
            raise ValueError("No target strikes pending")
        strike = self._pending_target_strikes[self._next_target_strike_index]
        plan = self._drone_plans[strike.drone_idx]
        success = action == AD_RESOLVE_ACTION_BASE + 1
        plan.strike_success = success
        target_idx = strike.target_idx
        already_destroyed = (
            0 <= target_idx < len(self._target_destroyed)
            and self._target_destroyed[target_idx]
        )
        if (
            success
            and 0 <= target_idx < len(self._targets)
            and not already_destroyed
        ):
            value = self._targets[target_idx].value
            plan.damage_inflicted += value
            self._damage_from_targets += value
            if 0 <= target_idx < len(self._target_destroyed):
                self._target_destroyed[target_idx] = True
        self._next_target_strike_index += 1
        if self._next_target_strike_index >= len(self._pending_target_strikes):
            self._pending_target_strikes.clear()
            self._next_target_strike_index = 0
            self._phase = Phase.TERMINAL
            self._finalize_returns()

    def _finalize_returns(self) -> None:
        total_damage = self._damage_from_targets
        self._returns = [total_damage, -total_damage]

    def is_terminal(self) -> bool:
        return self._phase == Phase.TERMINAL

    def returns(self) -> List[float]:
        return list(self._returns)

    def observation_string(self, player: int) -> str:
        lines = [f"Phase: {self._phase.name}"]
        lines.append("Targets:")
        for idx, target in enumerate(self._targets):
            lines.append(f"  T{idx}: ({target.row:.1f},{target.col:.1f}) value={target.value}")
        lines.append("AD units:")
        for idx, unit in enumerate(self._ad_units):
            status = "alive" if unit.alive else f"destroyed({unit.destroyed_by})"
            orient_deg = math.degrees(unit.orientation)
            lines.append(
                f"  AD{idx}: ({unit.row:.1f},{unit.col:.1f}) {status} orient={orient_deg:.0f}deg"
            )
        lines.append("Drones:")
        for idx, plan in enumerate(self._drone_plans):
            dest = self._describe_target(plan.target_idx)
            lines.append(
                f"  D{idx}: entry={plan.entry_point} -> {dest}"
                f" ToT={LARGE_TOT_CHOICES[plan.tot_idx]:.1f} strat={MIDPOINT_STRATEGIES[plan.blueprint_idx]}"
                f" destroyed_by={plan.destroyed_by}"
            )
        lines.append(f"Returns: {self._returns}")
        return "\n".join(lines)

    def information_state_string(self, player: int) -> str:
        return self.observation_string(player)

    def action_to_string(self, player: Optional[int], action: int) -> str:
        if TARGET_POSITION_ACTION_BASE <= action < TARGET_VALUE_ACTION_BASE:
            row, col = decode_target_position_action(action)
            return f"target_cell:({row:.1f},{col:.1f})"
        if TARGET_VALUE_ACTION_BASE <= action < AD_ACTION_BASE:
            perm = decode_target_value_action(action)
            return "target_values:" + ",".join(str(TARGET_VALUE_OPTIONS[idx]) for idx in perm)
        if AD_ACTION_BASE <= action < DRONE_ACTION_BASE:
            row, col = decode_ad_position_action(action)
            return f"ad_place:({row:.1f},{col:.1f})"
        if DRONE_ACTION_BASE <= action < INTERCEPT_ACTION_BASE:
            entry_idx, target_idx, tot_idx, blueprint_idx = decode_drone_action(action)
            target_desc = self._describe_target(target_idx)
            return (
                f"drone_assign:entry={entry_idx} target={target_desc}"
                f" ToT={LARGE_TOT_CHOICES[tot_idx]:.1f} strat={MIDPOINT_STRATEGIES[blueprint_idx]}"
            )
        if INTERCEPT_ACTION_BASE <= action < AD_RESOLVE_ACTION_BASE:
            choice = decode_interceptor_action(action)
            if choice is None:
                return "interceptor:pass"
            return f"interceptor:drone={choice}"
        if AD_RESOLVE_ACTION_BASE <= action < NUM_DISTINCT_ACTIONS:
            return "ad_resolution:success" if action == AD_RESOLVE_ACTION_BASE + 1 else "ad_resolution:fail"
        return f"unknown_action:{action}"

    def _describe_target(self, target_idx: int) -> str:
        if target_idx < len(self._targets):
            return f"target:{target_idx}"
        ad_idx = target_idx - len(self._targets)
        return f"ad:{ad_idx}"


_GAME_TYPE = pyspiel.GameType(
    short_name="swarm_defense_large",
    long_name="Swarm Defense Large-Scale Sequential Game",
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
    max_chance_outcomes=max(len(TARGET_CANDIDATE_CELLS), TARGET_VALUE_ACTIONS, 2),
    num_players=2,
    min_utility=-float(_MAX_DAMAGE),
    max_utility=float(_MAX_DAMAGE),
    utility_sum=0.0,
    max_game_length=(
        NUM_TARGETS
        + 1
        + NUM_AD_UNITS
        + NUM_ATTACKING_DRONES
        + NUM_INTERCEPTORS
        + NUM_AD_UNITS
    ),
)


class SwarmDefenseLargeGame(pyspiel.Game):
    def __init__(self, params: Optional[Dict[str, int]] = None):
        super().__init__(_GAME_TYPE, _GAME_INFO, params or {})

    def new_initial_state(self) -> SwarmDefenseLargeState:
        return SwarmDefenseLargeState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        return pyspiel.IIGObserverForPublicInfo(iig_obs_type, params)


pyspiel.register_game(_GAME_TYPE, SwarmDefenseLargeGame)

__all__ = [
    "SwarmDefenseLargeGame",
    "SwarmDefenseLargeState",
    "Phase",
    "TargetCluster",
    "ADUnit",
    "DronePlan",
    "decode_target_position_action",
    "decode_target_value_action",
    "decode_ad_position_action",
    "decode_drone_action",
    "encode_drone_action",
    "decode_interceptor_action",
    "LARGE_TOT_CHOICES",
    "ENTRY_POINTS",
    "MIDPOINT_STRATEGIES",
    "NUM_ATTACKING_DRONES",
    "NUM_INTERCEPTORS",
    "ARENA_WIDTH",
    "ARENA_HEIGHT",
    "AD_COVERAGE_RADIUS",
]
