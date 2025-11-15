"""OpenSpiel-aligned naive blueprint for the Swarm AD scenario."""

from __future__ import annotations

import importlib.util
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
SWARM_AD_OPEN_SPIEL_DIR = PROJECT_ROOT / "Swarm-AD-OpenSpiel"
if str(SWARM_AD_OPEN_SPIEL_DIR) not in sys.path:
    sys.path.append(str(SWARM_AD_OPEN_SPIEL_DIR))

Vec2 = Tuple[float, float]
GridCell = Tuple[int, int]


def _cell_to_position(cell: GridCell) -> Vec2:
    row, col = cell
    return (float(col), float(row))


# Constants mirrored from Swarm-AD-OpenSpiel/swarm_defense_game.py
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
TARGET_SPEED = 1.0
INTERCEPTOR_SPEED_MULTIPLIER = 2.0
INTERCEPTOR_LAUNCH_ROW = GRID_SIZE - 1
AD_KILL_RATE = 2.8
AD_MIN_EFFECTIVE_EXPOSURE = 0.75
INTERCEPTOR_KILL_PROB = 0.95
DRONE_VS_AD_KILL_PROB = 0.8
DRONE_VS_TARGET_KILL_PROB = 0.7
DEFAULT_BASELINE_SEED = 503721863

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

TARGET_CANDIDATE_CELLS: List[GridCell] = [
    (row, col)
    for row in range(BOTTOM_HALF_START, GRID_SIZE)
    for col in range(GRID_SIZE)
]
AD_POSITION_CANDIDATES: List[GridCell] = [
    (row, col)
    for row in range(BOTTOM_HALF_START, GRID_SIZE)
    if row % AD_STRIDE == 0
    for col in range(GRID_SIZE)
    if col % AD_STRIDE == 0
]


def _path_intersects(ad_pos: GridCell, entry: GridCell, target: GridCell) -> bool:
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
    ad_pos: GridCell,
    entry: GridCell,
    target: GridCell,
    samples: int = 80,
) -> Tuple[float, Tuple[float, float] | None, float | None]:
    if not _path_intersects(ad_pos, entry, target):
        return 0.0, None, None
    exposure = 0.0
    entry_point: Tuple[float, float] | None = None
    entry_distance: float | None = None
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


@dataclass(frozen=True)
class Target:
    name: str
    grid_cell: GridCell
    reward: float
    position: Vec2


@dataclass(frozen=True)
class Drone:
    name: str
    grid_cell: GridCell
    speed_cells_per_min: float
    position: Vec2


@dataclass(frozen=True)
class Interceptor:
    name: str
    grid_cell: GridCell
    speed_cells_per_min: float
    position: Vec2


@dataclass(frozen=True)
class AirDefense:
    name: str
    coverage_radius: float


@dataclass(frozen=True)
class Objective:
    name: str
    grid_cell: GridCell
    reward: float
    kind: str
    position: Vec2


@dataclass(frozen=True)
class AttackOrder:
    drone: Drone
    objective: Objective
    tot_choice: float
    travel_time: float
    time_on_target_min: float

    @property
    def objective_kind(self) -> str:
        return self.objective.kind


@dataclass(frozen=True)
class InterceptOrder:
    interceptor: Interceptor
    attack_order: AttackOrder
    intercept_time_min: float


@dataclass(frozen=True)
class AirDefensePlacement:
    ad_unit: AirDefense
    grid_cell: GridCell
    position: Vec2
    focus: Target


@dataclass(frozen=True)
class Scenario:
    drones: List[Drone]
    interceptors: List[Interceptor]
    targets: List[Target]
    ad_units: List[AirDefense]


@dataclass
class SimulationStats:
    intercept_sorties: int = 0
    interceptions_by_interceptors: int = 0
    ad_engagements: int = 0
    interceptions_by_ad: int = 0
    shots_on_targets: int = 0
    shots_on_ad: int = 0
    drones_reaching_targets: int = 0
    targets_destroyed: int = 0
    ad_units_destroyed: int = 0
    total_reward: float = 0.0
    surviving_drones: int = 0

    @property
    def total_interceptions(self) -> int:
        return self.interceptions_by_interceptors + self.interceptions_by_ad

    def summary_lines(self) -> List[str]:
        return [
            f"Reward delivered: {self.total_reward:.1f}",
            f"Target kills: {self.targets_destroyed}/{NUM_TARGETS}",
            f"Interceptions: {self.total_interceptions} (INT {self.interceptions_by_interceptors}, AD {self.interceptions_by_ad})",
            f"Shots on targets: {self.shots_on_targets} | on AD: {self.shots_on_ad}",
            f"Drones reaching target area: {self.surviving_drones}",
            f"AD units destroyed: {self.ad_units_destroyed}/{NUM_AD_UNITS}",
        ]


@dataclass(frozen=True)
class BlueprintOutcome:
    scenario: Scenario
    attack_orders: List[AttackOrder]
    intercept_orders: List[InterceptOrder]
    ad_placements: List[AirDefensePlacement]
    stats: SimulationStats


@dataclass(frozen=True)
class EpisodeStats:
    seed: int
    attacker_damage: float
    defender_reward: float
    ad_kills: int
    interceptor_kills: int
    ad_attrition: int
    survivors: int
    target_kills: int

    @property
    def total_intercepts(self) -> int:
        return self.ad_kills + self.interceptor_kills


@dataclass(frozen=True)
class EpisodeData:
    state: Any  # SwarmDefenseState at runtime
    snapshot: Dict[str, object]
    target_statuses: Tuple[Dict[str, Any], ...]
    stats: EpisodeStats


def create_reference_scenario(seed: int = 13) -> Scenario:
    rng = random.Random(seed)

    target_cells = rng.sample(TARGET_CANDIDATE_CELLS, NUM_TARGETS)
    value_perm = list(TARGET_VALUE_OPTIONS)
    rng.shuffle(value_perm)
    targets = [
        Target(
            name=f"Target-{idx+1}",
            grid_cell=cell,
            reward=value_perm[idx],
            position=_cell_to_position(cell),
        )
        for idx, cell in enumerate(target_cells)
    ]

    entry_columns = rng.sample(range(GRID_SIZE), NUM_ATTACKING_DRONES)
    drones = [
        Drone(
            name=f"DR-{idx:02d}",
            grid_cell=(0, col),
            speed_cells_per_min=TARGET_SPEED,
            position=_cell_to_position((0, col)),
        )
        for idx, col in enumerate(entry_columns)
    ]

    interceptor_columns = rng.sample(range(GRID_SIZE), NUM_INTERCEPTORS)
    interceptors = [
        Interceptor(
            name=f"INT-{idx:02d}",
            grid_cell=(INTERCEPTOR_LAUNCH_ROW, col),
            speed_cells_per_min=TARGET_SPEED * INTERCEPTOR_SPEED_MULTIPLIER,
            position=_cell_to_position((INTERCEPTOR_LAUNCH_ROW, col)),
        )
        for idx, col in enumerate(interceptor_columns)
    ]

    ad_units = [
        AirDefense(name=f"AD-{idx+1}", coverage_radius=AD_COVERAGE_RADIUS)
        for idx in range(NUM_AD_UNITS)
    ]

    return Scenario(drones=drones, interceptors=interceptors, targets=targets, ad_units=ad_units)


def choose_ad_placements(scenario: Scenario) -> List[AirDefensePlacement]:
    placements: List[AirDefensePlacement] = []
    occupied: set[GridCell] = set()
    prioritized_targets = sorted(scenario.targets, key=lambda t: t.reward, reverse=True)
    for ad_unit, target in zip(scenario.ad_units, prioritized_targets):
        best_cell: GridCell | None = None
        best_distance = float("inf")
        for cell in AD_POSITION_CANDIDATES:
            if cell in occupied:
                continue
            distance = math.dist(cell, target.grid_cell)
            if distance < best_distance:
                best_cell = cell
                best_distance = distance
        if best_cell is None:
            best_cell = prioritized_targets[0].grid_cell
        occupied.add(best_cell)
        placements.append(
            AirDefensePlacement(
                ad_unit=ad_unit,
                grid_cell=best_cell,
                position=_cell_to_position(best_cell),
                focus=target,
            )
        )
    return placements


def _proportional_allocation(total_items: int, weights: Sequence[float]) -> List[int]:
    if total_items < len(weights):
        raise ValueError("Need at least as many assets as objectives.")
    sum_w = sum(weights)
    if sum_w <= 0:
        return [total_items // len(weights) or 1 for _ in weights]
    raw = [total_items * (w / sum_w) for w in weights]
    counts = [math.floor(value) for value in raw]
    remainder = total_items - sum(counts)
    fractions = [value - math.floor(value) for value in raw]
    for idx in range(len(counts)):
        if counts[idx] == 0:
            counts[idx] = 1
            remainder -= 1
    while remainder < 0:
        candidates = [i for i, count in enumerate(counts) if count > 1]
        if not candidates:
            break
        idx = min(candidates, key=lambda i: weights[i])
        counts[idx] -= 1
        remainder += 1
    if remainder > 0:
        order = sorted(range(len(weights)), key=lambda i: fractions[i], reverse=True)
        for i in range(remainder):
            counts[order[i % len(order)]] += 1
    return counts


def build_attack_plan(
    scenario: Scenario,
    ad_placements: Sequence[AirDefensePlacement],
    seed: int | None = None,
) -> List[AttackOrder]:
    rng = random.Random(seed)

    objectives: List[Objective] = [
        Objective(
            name=target.name,
            grid_cell=target.grid_cell,
            reward=target.reward,
            kind="target",
            position=target.position,
        )
        for target in scenario.targets
    ]
    max_reward = max(target.reward for target in scenario.targets)
    ad_weight = max_reward * 0.4
    for placement in ad_placements:
        objectives.append(
            Objective(
                name=placement.ad_unit.name,
                grid_cell=placement.grid_cell,
                reward=0.0,
                kind="ad",
                position=placement.position,
            )
        )
    weights = [obj.reward if obj.kind == "target" else ad_weight for obj in objectives]
    counts = _proportional_allocation(len(scenario.drones), weights)

    if sum(counts) != len(scenario.drones):
        raise ValueError("Allocation mismatch with drone roster.")

    attack_orders: List[AttackOrder] = []
    drone_iter = iter(sorted(scenario.drones, key=lambda d: d.grid_cell[1]))
    for objective, objective_count in zip(objectives, counts):
        for _ in range(objective_count):
            drone = next(drone_iter)
            distance = math.dist(drone.grid_cell, objective.grid_cell)
            tot_choice = rng.choice(TOT_CHOICES)
            travel_time = distance / max(TARGET_SPEED, 1e-3)
            time_on_target = tot_choice + travel_time
            attack_orders.append(
                AttackOrder(
                    drone=drone,
                    objective=objective,
                    tot_choice=tot_choice,
                    travel_time=travel_time,
                    time_on_target_min=time_on_target,
                )
            )
    return attack_orders


def build_defense_plan(
    scenario: Scenario,
    attack_orders: Sequence[AttackOrder],
    seed: int | None = None,
) -> List[InterceptOrder]:
    rng = random.Random(seed)
    prioritized_attacks = sorted(
        attack_orders, key=lambda order: (order.objective.reward, -order.time_on_target_min), reverse=True
    )
    intercept_orders: List[InterceptOrder] = []
    for interceptor, attack_order in zip(scenario.interceptors, prioritized_attacks):
        intercept_offset = rng.uniform(0.5, 1.8)
        intercept_time = max(0.05, attack_order.time_on_target_min - intercept_offset)
        intercept_orders.append(
            InterceptOrder(
                interceptor=interceptor,
                attack_order=attack_order,
                intercept_time_min=intercept_time,
            )
        )
    return intercept_orders


def simulate_engagement(
    attack_orders: Sequence[AttackOrder],
    intercept_orders: Sequence[InterceptOrder],
    ad_placements: Sequence[AirDefensePlacement],
    seed: int | None = None,
) -> SimulationStats:
    rng = random.Random(seed)
    stats = SimulationStats()
    alive: Dict[AttackOrder, bool] = {order: True for order in attack_orders}

    stats.intercept_sorties = len(intercept_orders)
    for intercept in intercept_orders:
        order = intercept.attack_order
        if not alive.get(order, False):
            continue
    if rng.random() <= INTERCEPTOR_KILL_PROB:
            alive[order] = False
            stats.interceptions_by_interceptors += 1

    for placement in ad_placements:
        ad_cell = placement.grid_cell
        candidates: List[Tuple[float, float, AttackOrder]] = []
        for order in attack_orders:
            if not alive.get(order, False) or order.objective_kind == "ad":
                continue
            exposure, entry_point, entry_distance = _path_exposure_stats(
                ad_cell,
                order.drone.grid_cell,
                order.objective.grid_cell,
            )
            if exposure <= 0.0 or entry_point is None or entry_distance is None:
                continue
            intercept_time = order.tot_choice + entry_distance
            if intercept_time >= order.time_on_target_min:
                continue
            effective = max(exposure, AD_MIN_EFFECTIVE_EXPOSURE)
            probability = 1.0 - math.exp(-AD_KILL_RATE * effective)
            probability = max(0.0, min(1.0, probability))
            candidates.append((intercept_time, probability, order))
        if not candidates:
            continue
        candidates.sort(key=lambda item: (item[0], -item[1]))
        intercept_time, probability, selected_order = candidates[0]
        stats.ad_engagements += 1
        if rng.random() <= probability:
            alive[selected_order] = False
            stats.interceptions_by_ad += 1

    stats.surviving_drones = sum(1 for flag in alive.values() if flag)

    ad_destroyed: set[str] = set()
    hit_targets: set[str] = set()
    for order in attack_orders:
        if not alive.get(order, False):
            continue
        if order.objective_kind == "ad":
            stats.shots_on_ad += 1
            target_name = order.objective.name
            if target_name not in ad_destroyed and rng.random() <= DRONE_VS_AD_KILL_PROB:
                ad_destroyed.add(target_name)
                stats.ad_units_destroyed += 1
        else:
            stats.shots_on_targets += 1
            stats.drones_reaching_targets += 1
            if rng.random() <= DRONE_VS_TARGET_KILL_PROB:
                hit_targets.add(order.objective.name)
                stats.total_reward += order.objective.reward
        alive[order] = False
    stats.targets_destroyed = len(hit_targets)
    return stats


def run_blueprint(seed: int = 99) -> BlueprintOutcome:
    scenario = create_reference_scenario(seed=seed)
    ad_placements = choose_ad_placements(scenario)
    attack_orders = build_attack_plan(scenario, ad_placements, seed=seed + 3)
    intercept_orders = build_defense_plan(scenario, attack_orders, seed=seed + 5)
    stats = simulate_engagement(
        attack_orders=attack_orders,
        intercept_orders=intercept_orders,
        ad_placements=ad_placements,
        seed=seed + 9,
    )
    return BlueprintOutcome(
        scenario=scenario,
        attack_orders=attack_orders,
        intercept_orders=intercept_orders,
        ad_placements=ad_placements,
        stats=stats,
    )


def render_blueprint(outcome: BlueprintOutcome, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    rewards = [target.reward for target in outcome.scenario.targets]
    max_reward = max(rewards)
    for target in outcome.scenario.targets:
        size = 120 + 80 * (target.reward / max_reward)
        ax.scatter(*target.position, s=size, marker="*", color="#FFD166", edgecolor="#333333", zorder=4, label="Target")
        ax.text(
            target.position[0] + 0.2,
            target.position[1] + 0.6,
            f"{target.name}\nVal={target.reward:.0f}",
            fontsize=9,
            ha="left",
        )

    for order in outcome.attack_orders:
        drone_pos = order.drone.position
        target_pos = order.objective.position
        ax.scatter(*drone_pos, marker="^", color="#EF476F", s=40, zorder=3, label="Drone")
        ax.plot(
            (drone_pos[0], target_pos[0]),
            (drone_pos[1], target_pos[1]),
            color="#EF476F",
            linewidth=0.9,
            alpha=0.35,
        )
        mid_x = (drone_pos[0] + target_pos[0]) / 2
        mid_y = (drone_pos[1] + target_pos[1]) / 2
        ax.text(mid_x, mid_y, f"ToT {order.time_on_target_min:.1f}m", fontsize=7, color="#EF476F")

    for intercept in outcome.intercept_orders:
        pos = intercept.interceptor.position
        ax.scatter(*pos, marker="o", color="#118AB2", s=60, zorder=3, label="Interceptor")
        ax.text(
            pos[0] - 0.4,
            pos[1] - 0.4,
            intercept.interceptor.name,
            fontsize=8,
            color="#118AB2",
        )

    for placement in outcome.ad_placements:
        ax.scatter(
            *placement.position,
            marker="s",
            s=90,
            color="#06D6A0",
            edgecolor="#034732",
            linewidths=1.0,
            zorder=5,
            label="AD",
        )
        circle = plt.Circle(placement.position, AD_COVERAGE_RADIUS, color="#06D6A0", alpha=0.08)
        ax.add_patch(circle)
        ax.text(
            placement.position[0] - 0.6,
            placement.position[1] - 1.2,
            f"{placement.ad_unit.name}\nfocus {placement.focus.name}",
            fontsize=8,
            color="#034732",
        )

    ax.set_title("Naive Swarm Blueprint (OpenSpiel-aligned)")
    ax.set_xlim(-1, GRID_SIZE)
    ax.set_ylim(-1, GRID_SIZE)
    ax.set_xlabel("Grid East (cells)")
    ax.set_ylabel("Grid North (cells)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.4)

    handles, labels = ax.get_legend_handles_labels()
    unique: Dict[str, object] = {}
    for handle, label in zip(handles, labels):
        unique[label] = handle
    ax.legend(unique.values(), unique.keys(), loc="upper left")

    summary = "\n".join(outcome.stats.summary_lines())
    fig.text(0.02, 0.02, summary, fontsize=9, family="monospace", va="bottom")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    return output_path


# ---------------------------------------------------------------------------
# OpenSpiel-powered alignment helpers
# ---------------------------------------------------------------------------


def _compute_target_kill_status(
    drones: Tuple[Dict[str, object], ...], targets: Tuple[object, ...]
) -> Tuple[Dict[str, object], ...]:
    statuses: List[Dict[str, object]] = []
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
        if (not status["destroyed"]) or (arrival_time < (status["time"] or float("inf"))):
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


def _load_openspiel_demo_module():
    module_name = "swarm_ad_openspiel_demo_visualizer"
    if module_name in sys.modules:
        return sys.modules[module_name]
    module_path = SWARM_AD_OPEN_SPIEL_DIR / "demo_visualizer.py"
    if not module_path.exists():
        raise RuntimeError(
            "Could not locate Swarm-AD-OpenSpiel/demo_visualizer.py."
            " Ensure the repository layout matches the expected structure."
        )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the OpenSpiel demo visualizer module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    sys.modules[module_name] = module
    return module


def run_openspiel_episode(seed: Optional[int] = DEFAULT_BASELINE_SEED) -> EpisodeData:
    """Execute the actual Swarm Defense OpenSpiel game using the reference policies."""

    module = _load_openspiel_demo_module()

    try:
        state, used_seed = module.play_episode(seed)
    except ImportError as exc:  # pragma: no cover - pyspiel missing
        raise RuntimeError(
            "OpenSpiel (pyspiel) is not available. Run this script inside WSL or any"
            " environment where pyspiel is installed."
        ) from exc

    snapshot = state.snapshot()
    drones = snapshot["drones"]
    targets = snapshot["targets"]
    target_statuses = _compute_target_kill_status(drones, targets)
    ad_kills, interceptor_kills, survivors, ad_attrit = _count_outcomes(drones)
    returns = state.returns()
    stats = EpisodeStats(
        seed=used_seed,
        attacker_damage=float(returns[0]),
        defender_reward=float(returns[1]),
        ad_kills=ad_kills,
        interceptor_kills=interceptor_kills,
        ad_attrition=ad_attrit,
        survivors=survivors,
        target_kills=sum(1 for entry in target_statuses if entry.get("destroyed")),
    )
    return EpisodeData(
        state=state,
        snapshot=snapshot,
        target_statuses=target_statuses,
        stats=stats,
    )


def format_episode_summary(episode: EpisodeData) -> List[str]:
    stats = episode.stats
    return [
        f"Seed: {stats.seed}",
        f"Attacker damage: {stats.attacker_damage:.1f}",
        f"Defender reward: {stats.defender_reward:.1f}",
        f"Target kills: {stats.target_kills}/{NUM_TARGETS}",
        (
            "Interceptions: "
            f"{stats.total_intercepts} (AD {stats.ad_kills}, Interceptors {stats.interceptor_kills})"
        ),
        f"AD attrition shots: {stats.ad_attrition}",
        f"Drones that survived to strike area: {stats.survivors}",
    ]


def render_openspiel_episode(episode: EpisodeData, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    snapshot = episode.snapshot
    targets = snapshot["targets"]
    drones = snapshot["drones"]
    ad_units = snapshot["ad_units"]
    target_statuses = episode.target_statuses
    stats = episode.stats

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
    ax.imshow(grid, cmap="Greys", alpha=0.05, extent=(-0.5, GRID_SIZE - 0.5, GRID_SIZE - 0.5, -0.5))
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    ax.grid(True, which="both", color="lightgrey", linewidth=0.5, alpha=0.5)
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(GRID_SIZE - 0.5, -0.5)
    ax.set_title("Swarm Defense Blueprint on OpenSpiel")

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
    for row, col in AD_POSITION_CANDIDATES:
        ax.scatter(col, row, s=8, color="tab:blue", alpha=0.2)

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
        circle = plt.Circle((col, row), AD_COVERAGE_RADIUS, color=color, alpha=0.07)
        ax.add_patch(circle)

    ad_kill_label_added = False
    interceptor_kill_label_added = False
    for idx, drone in enumerate(drones):
        entry_col, entry_row = drone["entry"][1], drone["entry"][0]
        tgt_row, tgt_col = drone["destination"]
        tot = drone["tot"]
        color = TOT_PALETTE.get(tot, "tab:red")
        linestyle = "-" if drone["destroyed_by"] is None else "--"
        ax.plot([entry_col, tgt_col], [entry_row, tgt_row], color=color, linestyle=linestyle, linewidth=2)
        ax.scatter(entry_col, entry_row, color=color, marker="s", s=60)
        marker = "o"
        destroyed_by = (drone.get("destroyed_by") or "").lower()
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
                label = "AD intercept"
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
        if destroyed_by == "interceptor" and drone.get("interceptor_hit"):
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
                s=160,
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

    summary_lines = format_episode_summary(episode)
    summary_lines.append(
        f"Total intercepts expected: {stats.total_intercepts} (targeting 5 baseline)"
    )
    ax.text(
        0.02,
        0.04,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path
