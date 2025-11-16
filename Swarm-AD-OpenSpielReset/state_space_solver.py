"""State-space explorer for the Swarm Defense OpenSpiel game.

The tool can be pointed at any registered OpenSpiel game. Its goal is to
systematically expand the game tree, track how many nodes are visited, and report
traversal timing so we can sanity-check the size of the decision space prior to
running heavier algorithms such as ESCHER.

Example usage (from the repository root):

    python Swarm-AD-OpenSpiel/state_space_solver.py --max-nodes 5000 --progress 2

Pass --count-unique --chance-mode full to enumerate every reachable state and
report exact unique/terminal counts.

"""
from __future__ import annotations

import argparse
import importlib
import random
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Set, Tuple, Union

import pyspiel

try:  # Ensure the custom game is registered when the script runs locally.
    import swarm_defense_game  # noqa: F401
except ImportError:
    pass

@dataclass
class ExplorerConfig:
    traversal: str = "dfs"
    chance_mode: str = "full"
    max_depth: Optional[int] = None
    max_nodes: Optional[int] = None
    max_terminals: Optional[int] = None
    progress_interval_sec: Optional[float] = 5.0
    seed: Optional[int] = None
    time_limit_sec: Optional[float] = None
    count_unique_states: bool = False


@dataclass
class TraversalStats:
    total_nodes: int = 0
    decision_nodes: int = 0
    chance_nodes: int = 0
    terminal_nodes: int = 0
    unique_states: int = 0
    duplicate_nodes: int = 0
    branch_factor_sum: int = 0
    branching_samples: int = 0
    max_branching: int = 0
    max_depth_reached: int = 0
    depth_histogram: Counter = field(default_factory=Counter)
    player_node_counts: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    phase_counts: Counter = field(default_factory=Counter)
    unique_terminal_nodes: int = 0
    runtime_sec: float = 0.0
    stop_reason: Optional[str] = None

    @property
    def avg_branching(self) -> float:
        if self.branching_samples == 0:
            return 0.0
        return self.branch_factor_sum / self.branching_samples

    @property
    def nodes_per_second(self) -> float:
        if self.runtime_sec == 0:
            return float("inf")
        return self.total_nodes / self.runtime_sec


class StateSpaceExplorer:
    def __init__(self, game: pyspiel.Game, config: ExplorerConfig):
        self._game = game
        self._config = config
        self._rng = random.Random(config.seed)
        self._seen_states: Optional[Set[str]] = set() if config.count_unique_states else None

    def explore(self) -> TraversalStats:
        stats = TraversalStats()
        container: Union[
            Deque[Tuple[pyspiel.State, int]],
            List[Tuple[pyspiel.State, int]],
        ]
        if self._config.traversal == "bfs":
            container = deque()
            pop_fn = container.popleft
            push_fn = container.append
        else:
            container = []
            pop_fn = container.pop
            push_fn = container.append

        initial_state = self._game.new_initial_state()
        push_fn((initial_state, 0))
        start = time.perf_counter()
        next_progress = (
            start + self._config.progress_interval_sec
            if self._config.progress_interval_sec
            else None
        )

        while container:
            state, depth = pop_fn()

            if self._seen_states is not None:
                key = self._state_key(state)
                if key in self._seen_states:
                    stats.duplicate_nodes += 1
                    continue
                self._seen_states.add(key)
                stats.unique_states += 1

            self._record_state(stats, state, depth)

            if self._should_stop(stats):
                break

            if self._config.time_limit_sec is not None and (
                time.perf_counter() - start
            ) >= self._config.time_limit_sec:
                stats.stop_reason = "time_limit"
                break

            if self._config.max_depth is not None and depth >= self._config.max_depth:
                continue

            children = self._expand_children(state)
            if children:
                stats.branch_factor_sum += len(children)
                stats.branching_samples += 1
                stats.max_branching = max(stats.max_branching, len(children))

            for child in children:
                push_fn((child, depth + 1))

            if next_progress is not None and time.perf_counter() >= next_progress:
                self._print_progress(stats, time.perf_counter() - start)
                next_progress = time.perf_counter() + self._config.progress_interval_sec

        stats.runtime_sec = time.perf_counter() - start
        return stats

    def _state_key(self, state: pyspiel.State) -> str:
        try:
            return "ser:" + pyspiel.serialize_game_and_state(self._game, state)
        except (RuntimeError, ValueError):
            pass
        history = state.history_str()
        if history:
            return "hist:" + history
        try:
            obs = state.observation_string(0)
        except (RuntimeError, ValueError):
            obs = None
        if obs:
            return "obs:" + obs
        try:
            info = state.information_state_string(0)
        except (RuntimeError, ValueError):
            info = None
        if info:
            return "info:" + info
        return f"repr:{state}"

    def _record_state(self, stats: TraversalStats, state: pyspiel.State, depth: int) -> None:
        stats.total_nodes += 1
        stats.depth_histogram[depth] += 1
        stats.max_depth_reached = max(stats.max_depth_reached, depth)

        if state.is_terminal():
            stats.terminal_nodes += 1
            if self._config.count_unique_states:
                stats.unique_terminal_nodes += 1
        current_player = state.current_player()
        if current_player == pyspiel.PlayerId.CHANCE:
            stats.chance_nodes += 1
        elif current_player not in (pyspiel.PlayerId.TERMINAL, pyspiel.PlayerId.INVALID):
            stats.decision_nodes += 1
            stats.player_node_counts[current_player] += 1

        phase_name = self._extract_phase_name(state)
        if phase_name:
            stats.phase_counts[phase_name] += 1

    def _extract_phase_name(self, state: pyspiel.State) -> Optional[str]:
        phase_attr = getattr(state, "phase", None)
        if phase_attr is None:
            return None
        try:
            value = phase_attr() if callable(phase_attr) else phase_attr
        except TypeError:
            value = phase_attr
        if isinstance(value, Enum):
            return value.name
        if isinstance(value, str):
            return value
        return None

    def _expand_children(self, state: pyspiel.State) -> List[pyspiel.State]:
        if state.is_terminal():
            return []
        current_player = state.current_player()
        if current_player == pyspiel.PlayerId.CHANCE:
            outcomes = state.chance_outcomes()
            if not outcomes:
                return []
            if self._config.chance_mode == "sample":
                actions, probs = zip(*outcomes)
                choice = self._rng.choices(actions, weights=probs, k=1)[0]
                outcomes = [(choice, 1.0)]
            return [self._child_after_action(state, action) for action, _ in outcomes]
        if current_player == pyspiel.PlayerId.TERMINAL:
            return []
        actions = state.legal_actions()
        return [self._child_after_action(state, action) for action in actions]

    def _child_after_action(self, state: pyspiel.State, action: int) -> pyspiel.State:
        child = state.clone()
        child.apply_action(action)
        return child

    def _should_stop(self, stats: TraversalStats) -> bool:
        if self._config.max_nodes is not None and stats.total_nodes >= self._config.max_nodes:
            stats.stop_reason = stats.stop_reason or "max_nodes"
            return True
        if (
            self._config.max_terminals is not None
            and stats.terminal_nodes >= self._config.max_terminals
        ):
            stats.stop_reason = stats.stop_reason or "max_terminals"
            return True
        return False

    def _print_progress(self, stats: TraversalStats, elapsed: float) -> None:
        msg = f"[+{elapsed:6.1f}s] nodes={stats.total_nodes:,} terminals={stats.terminal_nodes:,}"
        if self._config.count_unique_states:
            msg += f" unique={stats.unique_states:,}"
        print(msg, file=sys.stderr)


def _format_depth_hist(depth_hist: Counter) -> str:
    if not depth_hist:
        return "(empty)"
    entries = sorted(depth_hist.items())
    preview = entries[:8]
    parts = [f"{depth}:{count:,}" for depth, count in preview]
    if len(entries) > len(preview):
        parts.append("…")
    return ", ".join(parts)


def _format_phase_counts(phase_counts: Counter) -> str:
    if not phase_counts:
        return "(phase attribute unavailable)"
    parts = [f"{name}:{count:,}" for name, count in phase_counts.most_common()]
    return ", ".join(parts)


def print_summary(stats: TraversalStats, config: ExplorerConfig, game_name: str) -> None:
    print("\n=== State-Space Exploration Summary ===")
    print(f"Game: {game_name}")
    print(f"Traversal: {config.traversal.upper()} | Chance: {config.chance_mode}")
    if config.max_depth is not None:
        print(f"Depth limit: {config.max_depth}")
    if config.max_nodes is not None:
        print(f"Node cap: {config.max_nodes:,}")
    if config.max_terminals is not None:
        print(f"Terminal cap: {config.max_terminals:,}")
    if config.time_limit_sec is not None:
        print(f"Time limit: {config.time_limit_sec:.1f}s")
    print()
    print(f"Total nodes expanded : {stats.total_nodes:,}")
    print(f"Decision nodes       : {stats.decision_nodes:,}")
    print(f"Chance nodes         : {stats.chance_nodes:,}")
    print(f"Terminal nodes       : {stats.terminal_nodes:,}")
    print(f"Max depth reached    : {stats.max_depth_reached}")
    if config.count_unique_states:
        print(f"Unique states        : {stats.unique_states:,}")
        print(f"Unique terminals     : {stats.unique_terminal_nodes:,}")
        print(f"Duplicates skipped   : {stats.duplicate_nodes:,}")
    print(f"Max branching factor : {stats.max_branching}")
    print(f"Avg branching factor : {stats.avg_branching:.2f}")
    print(f"Runtime (s)          : {stats.runtime_sec:.3f}")
    print(f"Nodes / second       : {stats.nodes_per_second:.2f}")
    print(f"Depth histogram      : {_format_depth_hist(stats.depth_histogram)}")
    print(f"Phase visits         : {_format_phase_counts(stats.phase_counts)}")
    if stats.player_node_counts:
        print("Player node counts   :", end=" ")
        parts = [f"P{player}:{count:,}" for player, count in sorted(stats.player_node_counts.items())]
        print(", ".join(parts))
    if stats.stop_reason:
        print(f"Stop reason          : {stats.stop_reason}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enumerate the state space for an OpenSpiel game.")
    parser.add_argument("--game", default="swarm_defense", help="Registered OpenSpiel game name.")
    parser.add_argument(
        "--import-module",
        action="append",
        default=[],
        help="Optional dotted modules to import before loading the game (to register custom games).",
    )
    parser.add_argument(
        "--traversal",
        choices=["dfs", "bfs"],
        default="dfs",
        help="Traversal strategy. DFS minimises memory, BFS keeps layers in order.",
    )
    parser.add_argument(
        "--chance-mode",
        choices=["full", "sample"],
        default="full",
        help="Enumerate all chance outcomes or sample one outcome per chance node.",
    )
    parser.add_argument("--max-depth", type=int, help="Optional depth limit (root depth=0).")
    parser.add_argument(
        "--max-nodes",
        type=int,
        help="Stop after visiting this many nodes (including the root).",
    )
    parser.add_argument(
        "--max-terminals",
        type=int,
        help="Stop after encountering this many terminal states.",
    )
    parser.add_argument(
        "--progress",
        type=float,
        default=5.0,
        help="Progress print interval in seconds (set to 0 to disable).",
    )
    parser.add_argument("--seed", type=int, help="Random seed used when --chance-mode=sample.")
    parser.add_argument(
        "--time-limit",
        type=float,
        help="Optional wall-clock time budget in seconds before stopping.",
    )
    parser.add_argument(
        "--count-unique",
        action="store_true",
        help="Hash states via history strings to deduplicate and report exact unique counts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for module in args.import_module:
        importlib.import_module(module)

    game = pyspiel.load_game(args.game)
    progress_interval = args.progress if args.progress and args.progress > 0 else None
    config = ExplorerConfig(
        traversal=args.traversal,
        chance_mode=args.chance_mode,
        max_depth=args.max_depth,
        max_nodes=args.max_nodes,
        max_terminals=args.max_terminals,
        progress_interval_sec=progress_interval,
        seed=args.seed,
        time_limit_sec=args.time_limit,
        count_unique_states=args.count_unique,
    )
    explorer = StateSpaceExplorer(game, config)
    stats = explorer.explore()
    print_summary(stats, config, args.game)


if __name__ == "__main__":
    main()
