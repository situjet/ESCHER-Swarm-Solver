"""Custom OpenSpiel game modeling a swarm attack vs. layered air defense."""
from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pyspiel

# Default game parameters (can be overridden via OpenSpiel game parameters)
_DEFAULT_GRID_ROWS = 16
_DEFAULT_GRID_COLS = 16
_DEFAULT_NUM_TARGETS = 3
_DEFAULT_NUM_AD_UNITS = 2
_DEFAULT_NUM_ATTACKING_DRONES = 10
_DEFAULT_NUM_INTERCEPTORS = 5
_DEFAULT_AD_KILL_PROBABILITY = 0.8  # Direct kill probability per unit of exposure
_DEFAULT_INTERCEPTOR_REWARD = 1.0

# Fixed game constants (not parameterizable)
TOT_CHOICES: Tuple[float, ...] = (0.0, 2.0, 4.0)
TARGET_VALUE_OPTIONS: Tuple[float, ...] = (10.0, 20.0, 40.0)
AD_COVERAGE_RADIUS = 2.0
AD_STRIDE = 2
TARGET_SPEED = 1.0
INTERCEPTOR_SPEED_MULTIPLIER = 2.0
AD_MIN_EFFECTIVE_EXPOSURE = 0.75
INTERCEPTOR_KILL_PROB = 0.95
DRONE_VS_AD_KILL_PROB = 0.8
DRONE_VS_TARGET_KILL_PROB = 0.7


@dataclass
class GameConfig:
    """Configuration for a SwarmDefense game instance."""
    grid_rows: int
    grid_cols: int
    num_targets: int
    num_ad_units: int
    num_attacking_drones: int
    num_interceptors: int
    ad_kill_probability: float  # Kill probability per unit of exposure (0.0 to 1.0)
    interceptor_reward: float
    
    @property
    def bottom_half_start(self) -> int:
        return self.grid_rows // 2
    
    @property
    def interceptor_launch_row(self) -> int:
        return self.grid_rows - 1
    
    @property
    def drone_target_slots(self) -> int:
        return self.num_targets + self.num_ad_units
    
    def get_target_candidate_cells(self) -> List[Tuple[int, int]]:
        """Get candidate cells for target placement.
        
        Excludes:
        - Front line (first row of bottom half)
        - Side borders (leftmost and rightmost columns)
        """
        return [
            (row, col)
            for row in range(self.bottom_half_start + 1, self.grid_rows)  # Skip front line
            for col in range(1, self.grid_cols - 1)  # Skip side borders (col 0 and last col)
        ]
    
    def get_ad_position_candidates(self) -> List[Tuple[int, int]]:
        return [
            (row, col)
            for row in range(self.bottom_half_start, self.grid_rows)
            if row % AD_STRIDE == 0
            for col in range(self.grid_cols)
            if col % AD_STRIDE == 0
        ]
    
    def get_target_value_permutations(self) -> List[Sequence[int]]:
        return list(itertools.permutations(range(self.num_targets)))
    
    def get_target_value_options(self) -> List[float]:
        """Return value options list sized to the number of targets."""
        if self.num_targets <= len(TARGET_VALUE_OPTIONS):
            return list(TARGET_VALUE_OPTIONS[: self.num_targets])
        # Extend by repeating the largest available value
        extended = list(TARGET_VALUE_OPTIONS)
        extended.extend([TARGET_VALUE_OPTIONS[-1]] * (self.num_targets - len(TARGET_VALUE_OPTIONS)))
        return extended
    
    def compute_action_space_info(self) -> Dict[str, int]:
        """Compute action space layout based on configuration."""
        target_cells = self.get_target_candidate_cells()
        value_perms = self.get_target_value_permutations()
        ad_positions = self.get_ad_position_candidates()
        
        target_position_actions = len(target_cells)
        target_value_actions = len(value_perms)
        ad_placement_actions = len(ad_positions)
        drone_assignment_actions = self.grid_cols * self.drone_target_slots * len(TOT_CHOICES)
        intercept_choices = self.num_attacking_drones + 1
        
        target_position_base = 0
        target_value_base = target_position_base + target_position_actions
        ad_base = target_value_base + target_value_actions
        drone_base = ad_base + ad_placement_actions
        intercept_base = drone_base + drone_assignment_actions
        ad_resolve_base = intercept_base + intercept_choices
        num_distinct = ad_resolve_base + 2
        
        return {
            "target_position_actions": target_position_actions,
            "target_value_actions": target_value_actions,
            "ad_placement_actions": ad_placement_actions,
            "drone_assignment_actions": drone_assignment_actions,
            "intercept_choices": intercept_choices,
            "target_position_base": target_position_base,
            "target_value_base": target_value_base,
            "ad_base": ad_base,
            "drone_base": drone_base,
            "intercept_base": intercept_base,
            "ad_resolve_base": ad_resolve_base,
            "num_distinct_actions": num_distinct,
        }
    
    def compute_max_game_length(self) -> int:
        """Worst-case number of actions (decisions + chance resolutions)."""
        setup_phases = (
            self.num_targets  # target positions
            + 1  # value permutation
            + self.num_ad_units  # AD placement
            + self.num_attacking_drones  # swarm assignment
            + self.num_interceptors  # interceptor assignment
        )
        # Resolution phases (upper bounds):
        interceptor_resolutions = self.num_interceptors
        ad_engagements = self.num_attacking_drones  # each drone can be engaged once
        drone_vs_ad_strikes = self.num_ad_units  # earliest attacker against each AD
        target_damage_resolutions = self.num_attacking_drones  # drones striking targets
        return (
            setup_phases
            + interceptor_resolutions
            + ad_engagements
            + drone_vs_ad_strikes
            + target_damage_resolutions
        )


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
    intercept_log: List[Tuple[int, Tuple[float, float], float]] = field(default_factory=list)


@dataclass
class DronePlan:
    entry_row: int
    entry_col: int
    target_idx: int
    tot_idx: int
    destroyed_by: Optional[str] = None
    intercepts: List[Tuple[int, Tuple[float, float], float]] = field(default_factory=list)
    interceptor_hit: Optional[Tuple[float, float]] = None
    interceptor_time: Optional[float] = None
    strike_success: Optional[bool] = None
    damage_inflicted: float = 0.0


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
) -> Tuple[float, Optional[Tuple[float, float]], Optional[float]]:
    if not _path_intersects(ad_pos, entry, target):
        return 0.0, None, None
    exposure = 0.0
    entry_point: Optional[Tuple[float, float]] = None
    entry_distance: Optional[float] = None
    steps = max(samples, 10)
    prev_point = (float(entry[0]), float(entry[1]))
    prev_inside = math.dist(prev_point, ad_pos) <= AD_COVERAGE_RADIUS
    if prev_inside:
        entry_point = prev_point
        entry_distance = 0.0
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
                entry_distance = math.dist((entry[0], entry[1]), entry_point)
        prev_point = point
        prev_inside = inside
    if entry_point is None:
        entry_point = point
        entry_distance = math.dist((entry[0], entry[1]), entry_point)
    if entry_distance is None:
        entry_distance = math.dist((entry[0], entry[1]), entry_point)
    return exposure, entry_point, entry_distance


def _interceptor_launch_point(plan: DronePlan, config: GameConfig) -> Tuple[float, float]:
    return (float(config.interceptor_launch_row), float(plan.entry_col))


def _compute_interceptor_intercept(
    plan: DronePlan,
    destination: Tuple[int, int],
    config: GameConfig,
    *,
    samples: int = 120,
) -> Optional[Tuple[float, Tuple[float, float]]]:
    entry = (float(plan.entry_row), float(plan.entry_col))
    launch = _interceptor_launch_point(plan, config)
    total_distance = math.dist(entry, destination)
    if total_distance <= 0.0:
        return None
    dest_arrival = _arrival_time_to_point(plan, destination)
    if dest_arrival <= 0.0:
        return None
    best_point: Optional[Tuple[float, float]] = None
    best_time: Optional[float] = None
    prev_s = 0.0
    for step in range(1, samples + 1):
        s = step / samples
        point = (
            entry[0] + (destination[0] - entry[0]) * s,
            entry[1] + (destination[1] - entry[1]) * s,
        )
        drone_time = TOT_CHOICES[plan.tot_idx] + total_distance * s
        interceptor_distance = math.dist(launch, point)
        interceptor_time = interceptor_distance / (TARGET_SPEED * INTERCEPTOR_SPEED_MULTIPLIER)
        gap = drone_time - interceptor_time
        if gap >= 0.0:
            # refine between previous point and current point for tighter intercept
            low = prev_s
            high = s
            for _ in range(8):
                mid = (low + high) / 2
                mid_point = (
                    entry[0] + (destination[0] - entry[0]) * mid,
                    entry[1] + (destination[1] - entry[1]) * mid,
                )
                mid_drone_time = TOT_CHOICES[plan.tot_idx] + total_distance * mid
                mid_interceptor_time = (
                    math.dist(launch, mid_point) / (TARGET_SPEED * INTERCEPTOR_SPEED_MULTIPLIER)
                )
                if mid_drone_time - mid_interceptor_time >= 0.0:
                    high = mid
                    point = mid_point
                    drone_time = mid_drone_time
                    interceptor_time = mid_interceptor_time
                else:
                    low = mid
            best_point = point
            best_time = min(drone_time, interceptor_time)
            break
    prev_s = s
    if best_point is None or best_time is None:
        return None
    if best_time >= dest_arrival:
        return None
    return best_time, best_point


class SwarmDefenseState(pyspiel.State):
    def __init__(self, game: "SwarmDefenseGame"):
        super().__init__(game)
        self._phase = Phase.TARGET_POSITIONS
        self._history: List[int] = []
        self._target_positions: List[Tuple[int, int]] = []
        self._targets: List[TargetCluster] = []
        self._target_destroyed: List[bool] = []
        self._ad_units: List[ADUnit] = []
        self._drone_plans: List[DronePlan] = []
        self._interceptor_steps = 0
        self._pending_ad_targets: List[ADIntercept] = []
        self._next_ad_resolution_index = 0
        self._pending_interceptor_hits: List[InterceptorEngagement] = []
        self._next_interceptor_resolution_index = 0
        self._pending_ad_strikes: List[DroneADStrike] = []
        self._next_ad_strike_index = 0
        self._pending_target_strikes: List[DroneTargetStrike] = []
        self._next_target_strike_index = 0
        self._damage_from_targets = 0.0
        self._interceptor_engaged: Set[int] = set()
        self._returns = [0.0, 0.0]

    def __deepcopy__(self, memo):
        """Optimized deepcopy for faster state cloning."""
        cls = self.__class__
        new_state = cls.__new__(cls)
        memo[id(self)] = new_state
        
        # Call parent init
        pyspiel.State.__init__(new_state, self.get_game())
        
        # Copy primitives directly
        new_state._phase = self._phase
        new_state._interceptor_steps = self._interceptor_steps
        new_state._next_ad_resolution_index = self._next_ad_resolution_index
        new_state._next_interceptor_resolution_index = self._next_interceptor_resolution_index
        new_state._next_ad_strike_index = self._next_ad_strike_index
        new_state._next_target_strike_index = self._next_target_strike_index
        new_state._damage_from_targets = self._damage_from_targets
        
        # Shallow copy lists of primitives/tuples
        new_state._history = self._history[:]
        new_state._target_positions = self._target_positions[:]
        new_state._target_destroyed = self._target_destroyed[:]
        new_state._returns = self._returns[:]
        
        # Copy dataclass lists - targets are frozen so shallow copy is fine
        new_state._targets = self._targets[:]
        
        # AD units and drone plans have nested lists, need proper copy
        new_state._ad_units = [
            ADUnit(u.row, u.col, u.alive, u.destroyed_by, u.intercept_log[:])
            for u in self._ad_units
        ]
        new_state._drone_plans = [
            DronePlan(p.entry_row, p.entry_col, p.target_idx, p.tot_idx, p.destroyed_by, 
                     p.intercepts[:], p.interceptor_hit, p.interceptor_time, p.strike_success, p.damage_inflicted)
            for p in self._drone_plans
        ]
        new_state._pending_ad_targets = self._pending_ad_targets[:]
        new_state._pending_interceptor_hits = self._pending_interceptor_hits[:]
        new_state._pending_ad_strikes = self._pending_ad_strikes[:]
        new_state._pending_target_strikes = self._pending_target_strikes[:]
        new_state._interceptor_engaged = self._interceptor_engaged.copy()
        
        return new_state

    @property
    def config(self) -> GameConfig:
        """Access the game configuration."""
        return self.get_game().config
    
    @property
    def action_info(self) -> Dict[str, int]:
        """Access action space information."""
        return self.get_game().action_info
    
    def phase(self) -> Phase:
        return self._phase
    
    def _decode_ad_position(self, action: int) -> Tuple[int, int]:
        """Decode AD placement action to grid position."""
        idx = action - self.action_info["ad_base"]
        ad_candidates = self.config.get_ad_position_candidates()
        if not (0 <= idx < len(ad_candidates)):
            raise ValueError("AD action outside candidate set")
        return ad_candidates[idx]
    
    def _decode_target_position_action(self, action: int) -> Tuple[int, int]:
        """Decode target placement action to grid position."""
        idx = action - self.action_info["target_position_base"]
        target_cells = self.config.get_target_candidate_cells()
        if not (0 <= idx < len(target_cells)):
            raise ValueError("Target action outside candidate set")
        return target_cells[idx]
    
    def _decode_target_value_action(self, action: int) -> Sequence[int]:
        """Decode target value permutation action."""
        idx = action - self.action_info["target_value_base"]
        perms = self.config.get_target_value_permutations()
        if not (0 <= idx < len(perms)):
            raise ValueError("Value permutation index out of bounds")
        return perms[idx]
    
    def _decode_drone_action(self, action: int) -> Tuple[int, int, int, int]:
        """Decode drone assignment action to (entry_row, entry_col, target_idx, tot_idx)."""
        rel = action - self.action_info["drone_base"]
        per_entry = self.config.drone_target_slots * len(TOT_CHOICES)
        entry_index = rel // per_entry
        entry_col = entry_index
        if not (0 <= entry_col < self.config.grid_cols):
            raise ValueError("Drone entry column out of bounds")
        target_index = (rel % per_entry) // len(TOT_CHOICES)
        tot_index = rel % len(TOT_CHOICES)
        return 0, entry_col, target_index, tot_index
    
    def _decode_interceptor_action(self, action: int) -> Optional[int]:
        """Decode interceptor action to drone index (or None for pass)."""
        rel = action - self.action_info["intercept_base"]
        if rel == self.config.num_attacking_drones:
            return None
        return rel

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
                    "intercept_log": tuple(unit.intercept_log),
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
                    "interceptor_hit": plan.interceptor_hit,
                    "interceptor_time": plan.interceptor_time,
                    "strike_success": plan.strike_success,
                    "damage_inflicted": plan.damage_inflicted,
                    "interceptor_engaged": idx in self._interceptor_engaged,
                }
                for idx, plan in enumerate(self._drone_plans)
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

    def _legal_actions(self, player: Optional[int] = None) -> List[int]:
        if self._phase == Phase.TERMINAL:
            return []
        if self._phase == Phase.TARGET_POSITIONS:
            target_cells = self.config.get_target_candidate_cells()
            remaining = [
                idx
                for idx, cell in enumerate(target_cells)
                if cell not in self._target_positions
            ]
            return [self.action_info["target_position_base"] + idx for idx in remaining]
        if self._phase == Phase.TARGET_VALUES:
            perms = self.config.get_target_value_permutations()
            return [
                self.action_info["target_value_base"] + idx
                for idx in range(len(perms))
            ]
        if self._phase == Phase.AD_PLACEMENT:
            occupied = {(unit.row, unit.col) for unit in self._ad_units}
            ad_candidates = self.config.get_ad_position_candidates()
            actions = []
            for idx, pos in enumerate(ad_candidates):
                if pos not in occupied:
                    actions.append(self.action_info["ad_base"] + idx)
            return actions
        if self._phase == Phase.SWARM_ASSIGNMENT:
            return [self.action_info["drone_base"] + i for i in range(self.action_info["drone_assignment_actions"])]
        if self._phase == Phase.INTERCEPT_ASSIGNMENT:
            choices = [
                self.action_info["intercept_base"] + idx
                for idx, plan in enumerate(self._drone_plans)
                if plan.destroyed_by is None and idx not in self._interceptor_engaged
            ]
            choices.append(self.action_info["intercept_base"] + self.config.num_attacking_drones)
            return choices
        if self._phase in (
            Phase.INTERCEPT_RESOLUTION,
            Phase.AD_RESOLUTION,
            Phase.DRONE_AD_STRIKE_RESOLUTION,
            Phase.TARGET_DAMAGE_RESOLUTION,
        ):
            return [self.action_info["ad_resolve_base"], self.action_info["ad_resolve_base"] + 1]
        return []

    def legal_actions(self, player: Optional[int] = None) -> List[int]:
        return self._legal_actions(player)

    def chance_outcomes(self) -> List[Tuple[int, float]]:
        if self._phase == Phase.TARGET_POSITIONS:
            target_cells = self.config.get_target_candidate_cells()
            remaining = [
                idx
                for idx, cell in enumerate(target_cells)
                if cell not in self._target_positions
            ]
            probability = 1.0 / len(remaining)
            return [
                (self.action_info["target_position_base"] + idx, probability)
                for idx in remaining
            ]
        if self._phase == Phase.TARGET_VALUES:
            perms = self.config.get_target_value_permutations()
            probability = 1.0 / len(perms)
            return [
                (self.action_info["target_value_base"] + idx, probability)
                for idx in range(len(perms))
            ]
        if self._phase == Phase.INTERCEPT_RESOLUTION:
            engagement = self._pending_interceptor_hits[self._next_interceptor_resolution_index]
            hit_prob = engagement.probability
            miss_prob = max(0.0, 1.0 - hit_prob)
            return [
                (self.action_info["ad_resolve_base"], miss_prob),
                (self.action_info["ad_resolve_base"] + 1, hit_prob),
            ]
        if self._phase == Phase.AD_RESOLUTION:
            intercept = self._pending_ad_targets[self._next_ad_resolution_index]
            hit_prob = intercept.probability
            miss_prob = max(0.0, 1.0 - hit_prob)
            return [
                (self.action_info["ad_resolve_base"], miss_prob),
                (self.action_info["ad_resolve_base"] + 1, hit_prob),
            ]
        if self._phase == Phase.DRONE_AD_STRIKE_RESOLUTION:
            strike = self._pending_ad_strikes[self._next_ad_strike_index]
            success = strike.probability
            failure = max(0.0, 1.0 - success)
            return [
                (self.action_info["ad_resolve_base"], failure),
                (self.action_info["ad_resolve_base"] + 1, success),
            ]
        if self._phase == Phase.TARGET_DAMAGE_RESOLUTION:
            strike = self._pending_target_strikes[self._next_target_strike_index]
            success = strike.probability
            failure = max(0.0, 1.0 - success)
            return [
                (self.action_info["ad_resolve_base"], failure),
                (self.action_info["ad_resolve_base"] + 1, success),
            ]
        return []

    def _apply_action(self, action: int) -> None:
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

    def apply_action(self, action: int) -> None:
        return self._apply_action(action)

    def _apply_target_position_action(self, action: int) -> None:
        cell = self._decode_target_position_action(action)
        if cell in self._target_positions:
            raise ValueError("Target cell already selected")
        self._target_positions.append(cell)
        if len(self._target_positions) == self.config.num_targets:
            self._phase = Phase.TARGET_VALUES

    def _apply_target_value_action(self, action: int) -> None:
        perm = self._decode_target_value_action(action)
        if len(self._target_positions) != self.config.num_targets:
            raise ValueError("Target positions incomplete")
        value_options = self.config.get_target_value_options()
        self._targets = [
            TargetCluster(
                row=self._target_positions[i][0],
                col=self._target_positions[i][1],
                value=value_options[perm[i]],
            )
            for i in range(self.config.num_targets)
        ]
        self._target_destroyed = [False] * len(self._targets)
        self._phase = Phase.AD_PLACEMENT

    def _apply_ad_action(self, action: int) -> None:
        row, col = self._decode_ad_position(action)
        if (row, col) in {(unit.row, unit.col) for unit in self._ad_units}:
            raise ValueError("AD position already occupied")
        self._ad_units.append(ADUnit(row=row, col=col))
        if len(self._ad_units) == self.config.num_ad_units:
            self._phase = Phase.SWARM_ASSIGNMENT

    def _apply_drone_action(self, action: int) -> None:
        if len(self._drone_plans) >= self.config.num_attacking_drones:
            raise ValueError("All drones already assigned")
        entry_row, entry_col, target_idx, tot_idx = self._decode_drone_action(action)
        max_target_index = len(self._targets) + len(self._ad_units)
        if not (0 <= target_idx < max_target_index):
            raise ValueError("Invalid target index for drone plan")
        plan = DronePlan(entry_row, entry_col, target_idx, tot_idx)
        self._drone_plans.append(plan)
        if len(self._drone_plans) == self.config.num_attacking_drones:
            self._phase = Phase.INTERCEPT_ASSIGNMENT

    def _apply_interceptor_action(self, action: int) -> None:
        if self._interceptor_steps >= self.config.num_interceptors:
            raise ValueError("No interceptors remaining")
        drone_idx = self._decode_interceptor_action(action)
        if drone_idx is not None:
            if not (0 <= drone_idx < len(self._drone_plans)):
                raise ValueError("Invalid drone index for interception")
            plan = self._drone_plans[drone_idx]
            if plan.destroyed_by is None and drone_idx not in self._interceptor_engaged:
                destination = self._drone_destination(plan)
                arrival_time = _arrival_time_to_point(plan, destination)
                intercept = _compute_interceptor_intercept(plan, destination, self.config)
                if intercept is not None:
                    intercept_time, intercept_point = intercept
                    if intercept_time < arrival_time:
                        self._pending_interceptor_hits.append(
                            InterceptorEngagement(
                                drone_idx=drone_idx,
                                hit_point=intercept_point,
                                intercept_time=intercept_time,
                            )
                        )
                        self._interceptor_engaged.add(drone_idx)
        self._interceptor_steps += 1
        if self._interceptor_steps == self.config.num_interceptors:
            self._start_post_interceptor_resolution()

    def _start_post_interceptor_resolution(self) -> None:
        if self._pending_interceptor_hits:
            self._phase = Phase.INTERCEPT_RESOLUTION
            self._next_interceptor_resolution_index = 0
        else:
            self._start_ad_resolution()

    def _apply_interceptor_resolution(self, action: int) -> None:
        if not self._pending_interceptor_hits:
            raise ValueError("No interceptor engagements pending")
        engagement = self._pending_interceptor_hits[self._next_interceptor_resolution_index]
        plan = self._drone_plans[engagement.drone_idx]
        success = action == self.action_info["ad_resolve_base"] + 1
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

    def _start_ad_resolution(self) -> None:
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
        if not self._pending_ad_targets:
            self._start_drone_ad_strike_resolution()
        else:
            self._phase = Phase.AD_RESOLUTION
            self._next_ad_resolution_index = 0

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
        success = action == self.action_info["ad_resolve_base"] + 1
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
        success = action == self.action_info["ad_resolve_base"] + 1
        plan.strike_success = success
        target_idx = strike.target_idx
        target_destroyed = (
            0 <= target_idx < len(self._target_destroyed)
            and self._target_destroyed[target_idx]
        )
        if (
            success
            and 0 <= target_idx < len(self._targets)
            and not target_destroyed
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
        best_payload: Optional[Tuple[int, Tuple[float, float], float, float]] = None
        for idx, plan in enumerate(self._drone_plans):
            if plan.destroyed_by is not None or idx in engaged:
                continue
            destination = self._drone_destination(plan)
            exposure, entry_point, entry_distance = _path_exposure_stats(
                position,
                (plan.entry_row, plan.entry_col),
                destination,
            )
            if exposure <= 0.0 or entry_point is None or entry_distance is None:
                continue
            intercept_time = TOT_CHOICES[plan.tot_idx] + entry_distance
            arrival_time = _arrival_time_to_point(plan, destination)
            if intercept_time >= arrival_time:
                continue
            distance_to_entry = math.dist(position, entry_point)
            sort_key = (distance_to_entry, intercept_time, idx)
            if best_sort is None or sort_key < best_sort:
                best_sort = sort_key
                best_payload = (idx, entry_point, intercept_time, exposure)
        if best_payload is None:
            return None
        drone_idx, entry_point, intercept_time, exposure = best_payload
        effective = max(exposure, AD_MIN_EFFECTIVE_EXPOSURE)
        # Simple linear model: probability scales with exposure, capped at 1.0
        probability = min(1.0, self.config.ad_kill_probability * effective)
        probability = max(0.0, min(1.0, probability))
        return ADIntercept(
            ad_idx=ad_idx,
            drone_idx=drone_idx,
            exposure=exposure,
            probability=probability,
            hit_point=entry_point,
            intercept_time=intercept_time,
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
        success = action == self.action_info["ad_resolve_base"] + 1
        arrival_time = _arrival_time_to_point(plan, self._drone_destination(plan))
        if (
            success
            and plan.destroyed_by is None
            and intercept.intercept_time < arrival_time
        ):
            plan.destroyed_by = f"ad:{intercept.ad_idx}"
            plan.intercepts.append((intercept.ad_idx, intercept.hit_point, intercept.intercept_time))
            ad_unit.intercept_log.append((intercept.drone_idx, intercept.hit_point, intercept.intercept_time))
        self._next_ad_resolution_index += 1
        if self._next_ad_resolution_index >= len(self._pending_ad_targets):
            self._pending_ad_targets.clear()
            self._next_ad_resolution_index = 0
            self._start_drone_ad_strike_resolution()

    def _finalize_returns(self) -> None:
        total_damage = self._damage_from_targets
        # Count interceptor kills and add reward for defender
        interceptor_kills = sum(1 for plan in self._drone_plans if plan.destroyed_by == "interceptor")
        interceptor_bonus = interceptor_kills * self.config.interceptor_reward
        # Attacker gets damage done minus interceptor penalty, defender gets opposite
        self._returns = [total_damage - interceptor_bonus, -total_damage + interceptor_bonus]

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

    def information_state_tensor(self, player: int) -> List[float]:
        tensor = []
        # Phase encoding (10 values for 10 phases)
        for p in Phase:
            tensor.append(1.0 if self._phase == p else 0.0)
        # Target positions (config.num_targets * 2 for row, col)
        for i in range(self.config.num_targets):
            if i < len(self._targets):
                # Use _targets if available (after TARGET_VALUES phase)
                tensor.extend([self._targets[i].row / self.config.grid_rows, self._targets[i].col / self.config.grid_cols])
            elif i < len(self._target_positions):
                # Use _target_positions if available (during TARGET_POSITIONS phase)
                pos = self._target_positions[i]
                tensor.extend([pos[0] / self.config.grid_rows, pos[1] / self.config.grid_cols])
            else:
                tensor.extend([0.0, 0.0])
        # Target values
        target_values = self.config.get_target_value_options()
        max_target_value = max(target_values) if target_values else 1.0
        for i in range(self.config.num_targets):
            if i < len(self._targets):
                tensor.append(self._targets[i].value / max_target_value)
            else:
                tensor.append(0.0)
        # AD units (config.num_ad_units * 3 for row, col, alive)
        for i in range(self.config.num_ad_units):
            if i < len(self._ad_units):
                unit = self._ad_units[i]
                tensor.extend([unit.row / self.config.grid_rows, unit.col / self.config.grid_cols, 1.0 if unit.alive else 0.0])
            else:
                tensor.extend([0.0, 0.0, 0.0])
        # Interceptor information (1 value for remaining interceptors; guard divide-by-zero)
        if self.config.num_interceptors > 0:
            remaining = float(self.config.num_interceptors - self._interceptor_steps)
            tensor.append(remaining / self.config.num_interceptors)
        else:
            tensor.append(0.0)
        # Drone plans (config.num_attacking_drones * 6 for entry_col, target_idx, tot_idx, alive, interceptor_assigned, destroyed_by_interceptor)
        for i in range(self.config.num_attacking_drones):
            if i < len(self._drone_plans):
                plan = self._drone_plans[i]
                interceptor_assigned = 1.0 if i in self._interceptor_engaged else 0.0
                destroyed_by_interceptor = 1.0 if plan.destroyed_by == "interceptor" else 0.0
                tensor.extend([
                    plan.entry_col / self.config.grid_cols,
                    plan.target_idx / self.config.drone_target_slots,
                    plan.tot_idx / len(TOT_CHOICES),
                    0.0 if plan.destroyed_by else 1.0,
                    interceptor_assigned,
                    destroyed_by_interceptor
                ])
            else:
                tensor.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        return tensor

    def observation_tensor(self, player: int) -> List[float]:
        """Return observation tensor for the given player.
        
        Since this is a perfect information game, observation is the same as information state.
        """
        return self.information_state_tensor(player)

    def action_to_string(self, player: Optional[int], action: int) -> str:
        target_pos_base = self.action_info["target_position_base"]
        target_val_base = self.action_info["target_value_base"]
        ad_base = self.action_info["ad_base"]
        drone_base = self.action_info["drone_base"]
        intercept_base = self.action_info["intercept_base"]
        ad_resolve_base = self.action_info["ad_resolve_base"]
        num_distinct = self.action_info["num_distinct_actions"]
        
        if target_pos_base <= action < target_val_base:
            row, col = self._decode_target_position_action(action)
            return f"target_cell:({row},{col})"
        if target_val_base <= action < ad_base:
            perm = self._decode_target_value_action(action)
            values = self.config.get_target_value_options()
            return "target_values:" + ",".join(str(values[idx]) for idx in perm)
        if ad_base <= action < drone_base:
            row, col = self._decode_ad_position(action)
            return f"ad_place:({row},{col})"
        if drone_base <= action < intercept_base:
            entry_row, entry_col, target_idx, tot_idx = self._decode_drone_action(action)
            target_desc = self._describe_target_type(target_idx)
            return (
                f"drone_assign:entry=({entry_row},{entry_col}) target={target_desc}"
                f" ToT={TOT_CHOICES[tot_idx]}"
            )
        if intercept_base <= action < ad_resolve_base:
            choice = self._decode_interceptor_action(action)
            if choice is None:
                return "interceptor:pass"
            return f"interceptor:drone={choice}"
        if ad_resolve_base <= action < num_distinct:
            return "ad_resolution:success" if action == ad_resolve_base + 1 else "ad_resolution:fail"
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
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={
        "grid_rows": _DEFAULT_GRID_ROWS,
        "grid_cols": _DEFAULT_GRID_COLS,
        "num_targets": _DEFAULT_NUM_TARGETS,
        "num_ad_units": _DEFAULT_NUM_AD_UNITS,
        "num_attacking_drones": _DEFAULT_NUM_ATTACKING_DRONES,
        "num_interceptors": _DEFAULT_NUM_INTERCEPTORS,
        "ad_kill_probability": _DEFAULT_AD_KILL_PROBABILITY,
        "interceptor_reward": _DEFAULT_INTERCEPTOR_REWARD,
    },
)


class SwarmDefenseGame(pyspiel.Game):
    """Swarm Defense game with configurable parameters."""
    
    def __init__(self, params: Optional[Dict] = None):
        # Extract parameters with defaults
        params = params or {}
        self.config = GameConfig(
            grid_rows=int(params.get("grid_rows", _DEFAULT_GRID_ROWS)),
            grid_cols=int(params.get("grid_cols", _DEFAULT_GRID_COLS)),
            num_targets=int(params.get("num_targets", _DEFAULT_NUM_TARGETS)),
            num_ad_units=int(params.get("num_ad_units", _DEFAULT_NUM_AD_UNITS)),
            num_attacking_drones=int(params.get("num_attacking_drones", _DEFAULT_NUM_ATTACKING_DRONES)),
            num_interceptors=int(params.get("num_interceptors", _DEFAULT_NUM_INTERCEPTORS)),
            ad_kill_probability=float(
                params.get("ad_kill_probability") if "ad_kill_probability" in params
                else (1.0 - math.exp(-params.get("ad_kill_rate", 2.8) * AD_MIN_EFFECTIVE_EXPOSURE)) if "ad_kill_rate" in params
                else _DEFAULT_AD_KILL_PROBABILITY
            ),
            interceptor_reward=float(params.get("interceptor_reward", _DEFAULT_INTERCEPTOR_REWARD)),
        )
        
        # Compute action space and game info based on configuration
        self.action_info = self.config.compute_action_space_info()
        
        # Compute max utility
        max_damage = self.config.num_attacking_drones * max(TARGET_VALUE_OPTIONS)
        max_interceptor_bonus = self.config.num_attacking_drones * self.config.interceptor_reward
        max_utility = max_damage + max_interceptor_bonus
        
        # Compute max chance outcomes
        target_cells = self.config.get_target_candidate_cells()
        value_perms = self.config.get_target_value_permutations()
        max_chance_outcomes = max(len(target_cells), len(value_perms), 2)
        
        # Create game info with configuration-specific values
        game_info = pyspiel.GameInfo(
            num_distinct_actions=self.action_info["num_distinct_actions"],
            max_chance_outcomes=max_chance_outcomes,
            num_players=2,
            min_utility=-float(max_utility),
            max_utility=float(max_utility),
            utility_sum=0.0,
            max_game_length=self.config.compute_max_game_length(),
        )
        
        super().__init__(_GAME_TYPE, game_info, params or {})

    def new_initial_state(self) -> SwarmDefenseState:
        return SwarmDefenseState(self)


pyspiel.register_game(_GAME_TYPE, SwarmDefenseGame)

__all__ = [
    "SwarmDefenseGame",
    "SwarmDefenseState",
    "GameConfig",
    "TargetCluster",
    "ADUnit",
    "DronePlan",
    "Phase",
    "TOT_CHOICES",
]
