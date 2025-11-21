"""Policy inference and visualization helper for Swarm Defense Large v2.

This script loads a trained ESCHER-Torch policy (produced by
run_escher_torch_swarm_2.py), rolls out one or more evaluation episodes, and
saves visual artifacts (PNG snapshot + GIF animation) alongside structured
summaries of the episode returns.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pyspiel  # type: ignore
import torch

# Force a non-interactive Matplotlib backend before importing the demo helpers.
os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parent
SWARM_REPO_PATH = PROJECT_ROOT.parent / "Swarm-AD-Large-OpenSpiel-2"
if str(SWARM_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(SWARM_REPO_PATH))

from demo_animation import _build_animation as build_animation, _write_snapshot as write_snapshot  # type: ignore
from demo_visualizer import render_snapshot  # type: ignore
from swarm_defense_large_game import (
    DRONE_ACTION_BASE,
    DRONE_ALLOC_ACTION_BASE,
    INTERCEPT_ACTION_BASE,
    INTERCEPT_ALLOC_ACTION_BASE,
    INTERCEPT_END_ACTION,
    Phase,
    SwarmDefenseLargeState,
)  # type: ignore

from ESCHER_Torch.eschersolver import PolicyNetwork
from run_escher_torch_swarm_2 import (
    GAME_NAME,
    NETWORK_LAYERS,
    RESULT_DIR_NAME,
    ensure_information_state_tensor,
)

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / RESULT_DIR_NAME / "inference_runs"


@dataclass
class EpisodeArtifacts:
    scenario: str
    seed: int
    steps: int
    returns: Tuple[float, float]
    snapshot_path: Optional[Path]
    animation_path: Optional[Path]
    wintak_snapshot_path: Optional[Path]
    summary_path: Path


@dataclass
class ScenarioSpec:
    name: str
    description: str
    controller_factory: Callable[[], Dict[int, "Controller"]]


class Controller(ABC):
    @abstractmethod
    def choose_action(self, state: pyspiel.State, rng: random.Random) -> Tuple[int, Dict[str, object]]:
        """Return an action and optional metadata for logging."""


class PolicyController(Controller):
    """Wraps a trained PolicyNetwork for action selection."""

    def __init__(self, game: pyspiel.Game, policy_path: Path, *, device: str = "cpu", sampling: bool = False) -> None:
        ensure_information_state_tensor()
        self.game = game
        self.device = device
        self.sampling = sampling
        template_state = self.game.new_initial_state()
        self.info_dim = len(template_state.information_state_tensor(0))
        self.num_actions = self.game.num_distinct_actions()
        self.network = PolicyNetwork(
            input_size=self.info_dim,
            hidden_layers=NETWORK_LAYERS,
            num_actions=self.num_actions,
        )
        state_dict = torch.load(str(policy_path), map_location=device)
        self.network.load_state_dict(state_dict)
        self.network.eval()
        self.network.to(device)

    def choose_action(self, state: pyspiel.State, rng: random.Random) -> Tuple[int, Dict[str, object]]:
        player = state.current_player()
        legal_actions = state.legal_actions(player)
        if not legal_actions:
            raise RuntimeError(f"No legal actions for player {player}")
        if len(legal_actions) == 1:
            action = legal_actions[0]
            return action, {"controller": "policy"}
        info_state = np.asarray(state.information_state_tensor(player), dtype=np.float32)
        mask = _legal_mask(state, player, self.num_actions)
        info_tensor = torch.from_numpy(info_state).unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.network(info_tensor, mask_tensor)[0].cpu().numpy()
        action_probabilities = {action: float(max(probs[action], 0.0)) for action in legal_actions}
        if self.sampling:
            weights = [action_probabilities[action] for action in legal_actions]
            weights = [w if w > 0.0 else 1e-9 for w in weights]
            action = rng.choices(legal_actions, weights=weights, k=1)[0]
        else:
            action = max(legal_actions, key=lambda a: action_probabilities.get(a, 0.0))

        metadata: Dict[str, object] = {"probabilities": action_probabilities, "controller": "policy"}
        phase = state.phase() if hasattr(state, "phase") else None
        # Mirror the small-game fix: discourage premature "end wave" selections when
        # interceptors are still available and drones remain to target.
        if (
            player == 1
            and phase == Phase.INTERCEPT_ASSIGNMENT
            and action == INTERCEPT_END_ACTION
        ):
            snapshot = state.snapshot() if hasattr(state, "snapshot") else {}
            active = int(snapshot.get("wave_interceptors_active", 0))
            drone_actions = [a for a in legal_actions if a != INTERCEPT_END_ACTION]
            if active > 0 and drone_actions:
                best_drone = max(drone_actions, key=lambda a: action_probabilities.get(a, 0.0))
                if best_drone != action:
                    metadata["override"] = "intercept_drone"
                    metadata["override_from"] = action
                    metadata["override_to"] = best_drone
                action = best_drone
        return action, metadata


class NaiveAttackerController(Controller):
    def __init__(self, *, wave_budgets: Sequence[int] = (5, 5), label: str = "naive_attacker_10_per_wave") -> None:
        self.wave_budgets = {idx + 1: budget for idx, budget in enumerate(wave_budgets)}
        self.label = label
        self._launched_by_wave: Dict[int, int] = {}

    def _record_launch(self, wave: int) -> None:
        self._launched_by_wave[wave] = self._launched_by_wave.get(wave, 0) + 1

    def choose_action(self, state: pyspiel.State, rng: random.Random) -> Tuple[int, Dict[str, object]]:
        player = state.current_player()
        if player != 0:
            raise RuntimeError("NaiveAttackerController invoked for non-attacker player")
        legal_actions = state.legal_actions(player)
        if not legal_actions:
            raise RuntimeError("No legal attacker actions available")
        if len(legal_actions) == 1:
            return legal_actions[0], {"controller": self.label}
        phase = state.phase() if hasattr(state, "phase") else None
        if phase == Phase.SWARM_WAVE_ALLOCATION:
            snapshot = state.snapshot()
            current_wave = int(snapshot.get("current_wave", 1))
            remaining = int(snapshot.get("remaining_drones", 0))
            desired = self.wave_budgets.get(current_wave, remaining)
            desired = max(0, min(desired, remaining))
            allocations = sorted(action - DRONE_ALLOC_ACTION_BASE for action in legal_actions)
            if allocations:
                desired = max(allocations[0], min(desired, allocations[-1]))
                self.wave_budgets[current_wave] = desired
                action = DRONE_ALLOC_ACTION_BASE + desired
                if action in legal_actions:
                    return action, {"controller": self.label, "allocation": desired}
            return rng.choice(legal_actions), {"controller": self.label}

        if phase == Phase.SWARM_ASSIGNMENT:
            snapshot = state.snapshot()
            current_wave = int(snapshot.get("current_wave", 1))
            drone_actions = [a for a in legal_actions if a >= DRONE_ACTION_BASE]
            if drone_actions:
                action = rng.choice(drone_actions)
                self._record_launch(current_wave)
                return action, {"controller": self.label}
        return rng.choice(legal_actions), {"controller": self.label}


class NaiveDefenderController(Controller):
    def __init__(self, *, intercepts_per_wave: int = 10, label: str = "naive_defender_10_per_wave") -> None:
        self.intercepts_per_wave = intercepts_per_wave
        self.label = label
        self._shots_fired: Dict[int, int] = {}

    def choose_action(self, state: pyspiel.State, rng: random.Random) -> Tuple[int, Dict[str, object]]:
        player = state.current_player()
        if player != 1:
            raise RuntimeError("NaiveDefenderController invoked for non-defender player")
        legal_actions = state.legal_actions(player)
        if not legal_actions:
            raise RuntimeError("No legal defender actions available")
        if len(legal_actions) == 1:
            return legal_actions[0], {"controller": self.label}
        phase = state.phase() if hasattr(state, "phase") else None
        if phase == Phase.INTERCEPTOR_WAVE_ALLOCATION:
            snapshot = state.snapshot()
            current_wave = int(snapshot.get("current_wave", 1))
            remaining = int(snapshot.get("remaining_interceptors", 0))
            desired = max(0, min(self.intercepts_per_wave, remaining))
            allocations = sorted(action - INTERCEPT_ALLOC_ACTION_BASE for action in legal_actions)
            if allocations:
                desired = max(allocations[0], min(desired, allocations[-1]))
                self._shots_fired[current_wave] = 0
                action = INTERCEPT_ALLOC_ACTION_BASE + desired
                if action in legal_actions:
                    return action, {"controller": self.label, "allocation": desired}
            return rng.choice(legal_actions), {"controller": self.label}

        if phase == Phase.INTERCEPT_ASSIGNMENT:
            snapshot = state.snapshot()
            current_wave = int(snapshot.get("current_wave", 1))
            cap = min(self.intercepts_per_wave, int(snapshot.get("wave_interceptors_active", 0)))
            fired = self._shots_fired.get(current_wave, 0)
            intercept_actions = [a for a in legal_actions if INTERCEPT_ACTION_BASE <= a < INTERCEPT_END_ACTION]
            if cap <= 0 or fired >= cap or not intercept_actions:
                if INTERCEPT_END_ACTION in legal_actions:
                    return INTERCEPT_END_ACTION, {"controller": self.label}
                return rng.choice(legal_actions), {"controller": self.label}
            choice = rng.choice(intercept_actions)
            self._shots_fired[current_wave] = fired + 1
            return choice, {"controller": self.label}
        return rng.choice(legal_actions), {"controller": self.label}


def _sample_chance_action(state: pyspiel.State, rng: random.Random) -> int:
    outcomes = state.chance_outcomes()
    if not outcomes:
        raise RuntimeError("Chance node without outcomes encountered")
    pick = rng.random()
    cumulative = 0.0
    for action, prob in outcomes:
        cumulative += prob
        if pick <= cumulative:
            return action
    return outcomes[-1][0]


def _legal_mask(state: pyspiel.State, player: int, num_actions: int) -> np.ndarray:
    mask_fn = getattr(state, "legal_actions_mask", None)
    if callable(mask_fn):
        mask = np.asarray(mask_fn(player), dtype=np.float32)
        if mask.shape[0] == num_actions:
            return mask
    mask = np.zeros(num_actions, dtype=np.float32)
    for action in state.legal_actions(player):
        mask[action] = 1.0
    return mask


def _top_k(probabilities: Dict[int, float], k: int = 5) -> List[Tuple[int, float]]:
    return sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:k]


def run_episode(
    game: pyspiel.Game,
    controllers: Dict[int, Controller],
    seed: int,
) -> Tuple[SwarmDefenseLargeState, List[Dict[str, object]], Tuple[float, float], int]:
    rng = random.Random(seed)
    state = game.new_initial_state()
    decision_trace: List[Dict[str, object]] = []
    steps = 0

    while not state.is_terminal():
        player = state.current_player()
        if player == pyspiel.PlayerId.CHANCE:
            action = _sample_chance_action(state, rng)
            decision_trace.append({"step": steps, "player": "chance", "action": action})
        else:
            controller = controllers.get(player)
            if controller is None:
                raise ValueError(f"No controller configured for player {player}")
            action, meta = controller.choose_action(state, rng)
            entry: Dict[str, object] = {"step": steps, "player": int(player), "action": action}
            if meta:
                probs = meta.get("probabilities")
                if isinstance(probs, dict):
                    entry["top_actions"] = _top_k(probs)
                controller_name = meta.get("controller")
                if controller_name:
                    entry["controller"] = controller_name
            decision_trace.append(entry)
        state.apply_action(action)
        steps += 1

    returns = tuple(float(value) for value in state.returns())
    assert isinstance(state, SwarmDefenseLargeState)
    return state, decision_trace, returns, steps


def _write_summary(
    episode_dir: Path,
    seed: int,
    steps: int,
    returns: Sequence[float],
    decision_trace: Sequence[Dict[str, object]],
    snapshot_path: Optional[Path],
    animation_path: Optional[Path],
    wintak_snapshot_path: Optional[Path],
) -> Path:
    payload = {
        "seed": seed,
        "steps": steps,
        "returns": {
            "attacker": float(returns[0]) if len(returns) > 0 else 0.0,
            "defender": float(returns[1]) if len(returns) > 1 else 0.0,
        },
        "snapshot_path": str(snapshot_path) if snapshot_path else None,
        "animation_path": str(animation_path) if animation_path else None,
        "wintak_snapshot_path": str(wintak_snapshot_path) if wintak_snapshot_path else None,
        "decision_trace": decision_trace,
    }
    summary_path = episode_dir / "episode_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return summary_path


def _resolve_checkpoint_dir(explicit: Optional[Path]) -> Path:
    def candidate_dirs(root: Path) -> Iterable[Path]:
        if not root.exists():
            return []
        if (root / "policy.pt").exists():
            return [root]
        subdirs = [child for child in root.iterdir() if child.is_dir()] if root.is_dir() else []
        return sorted(subdirs, key=lambda path: path.stat().st_mtime, reverse=True)

    search_roots: List[Path] = []
    if explicit is not None:
        search_roots.append(explicit)
    default_root = PROJECT_ROOT / "results" / RESULT_DIR_NAME
    search_roots.append(default_root)
    search_roots.append(PROJECT_ROOT)

    visited: Set[Path] = set()
    for root in search_roots:
        for candidate in candidate_dirs(root):
            if candidate in visited:
                continue
            visited.add(candidate)
            policy_path = candidate / "policy.pt"
            if policy_path.exists():
                return candidate
    raise FileNotFoundError(
        "Unable to locate a checkpoint directory containing policy.pt. "
        "Use --checkpoint to point at a folder produced by run_escher_torch_swarm_2.py."
    )


def _episode_output_dir(base: Path, episode_idx: int, seed: int) -> Path:
    return base / f"episode_{episode_idx:02d}_seed_{seed}"


def _save_visuals(
    state: SwarmDefenseLargeState,
    episode_dir: Path,
    *,
    make_snapshot: bool,
    make_animation: bool,
    export_wintak_snapshot: bool,
    time_step: float,
    fps: int,
) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    snapshot_path = episode_dir / "snapshot.png" if make_snapshot else None
    animation_path = episode_dir / "animation.gif" if make_animation else None
    wintak_snapshot_path: Optional[Path] = None
    if snapshot_path is not None:
        render_snapshot(state, snapshot_path)
    if animation_path is not None:
        build_animation(state, animation_path, time_step=time_step, fps=fps)
    if export_wintak_snapshot:
        wintak_snapshot_path = episode_dir / "wintak_snapshot.json"
        write_snapshot(state.snapshot(), wintak_snapshot_path)
    return snapshot_path, animation_path, wintak_snapshot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with a trained ESCHER-Torch policy on Swarm Defense Large v2",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Directory containing policy.pt (defaults to the latest results folder)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to load the policy on",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Episodes per scenario (triad mode runs this many for each matchup)",
    )
    parser.add_argument(
        "--scenario-mode",
        type=str,
        choices=["triad", "single"],
        default="triad",
        help="single=policy vs policy only, triad=three preset matchups",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base RNG seed (each episode increments the seed)",
    )
    parser.add_argument(
        "--sampling",
        action="store_true",
        help="Sample from the policy distribution instead of taking argmax",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory to store snapshots/animations/summaries",
    )
    parser.add_argument(
        "--no-snapshot",
        action="store_true",
        help="Skip writing the PNG snapshot",
    )
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="Skip writing the GIF animation",
    )
    parser.add_argument(
        "--no-wintak-snapshot",
        action="store_true",
        help="Skip exporting the JSON snapshot used by the WinTAK/CoT pipeline",
    )
    parser.add_argument(
        "--time-step",
        type=float,
        default=0.25,
        help="Simulation timestep for animations (seconds)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=12,
        help="Frames per second for animations",
    )
    return parser.parse_args()


def _build_scenarios(mode: str, policy_controller: PolicyController) -> List[ScenarioSpec]:
    if mode == "single":
        return [
            ScenarioSpec(
                name="policy_vs_policy",
                description="Player 0 and Player 1 use the trained policy",
                controller_factory=lambda pc=policy_controller: {0: pc, 1: pc},
            )
        ]

    return [
        ScenarioSpec(
            name="player1_policy_vs_naive_attacker",
            description="Player 1 optimal policy vs naive attacker (10 drones per wave)",
            controller_factory=lambda pc=policy_controller: {
                0: NaiveAttackerController(),
                1: pc,
            },
        ),
        ScenarioSpec(
            name="player0_policy_vs_naive_defender",
            description="Player 0 optimal policy vs naive defender (10 intercepts per wave)",
            controller_factory=lambda pc=policy_controller: {
                0: pc,
                1: NaiveDefenderController(),
            },
        ),
        ScenarioSpec(
            name="policy_vs_policy",
            description="Both players use the trained policy",
            controller_factory=lambda pc=policy_controller: {0: pc, 1: pc},
        ),
    ]


def main() -> None:
    args = parse_args()
    checkpoint_dir = _resolve_checkpoint_dir(args.checkpoint)
    policy_path = checkpoint_dir / "policy.pt"
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy checkpoint not found: {policy_path}")

    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    base_seed = args.seed if args.seed is not None else random.SystemRandom().randint(0, 2**31 - 1)

    print("Swarm Defense Large v2 | Policy Inference")
    print(f"Checkpoint: {checkpoint_dir}")
    print(
        "Device: {device} | Scenario mode: {mode} | Episodes/scenario: {episodes} | Sampling: {sampling}".format(
            device=args.device, mode=args.scenario_mode, episodes=args.episodes, sampling=args.sampling
        )
    )
    print(f"Outputs → {output_root}")

    ensure_information_state_tensor()
    game = pyspiel.load_game(GAME_NAME)
    policy_controller = PolicyController(game, policy_path, device=args.device, sampling=args.sampling)
    scenarios = _build_scenarios(args.scenario_mode, policy_controller)
    artifacts: List[EpisodeArtifacts] = []

    for scenario_idx, spec in enumerate(scenarios, start=1):
        scenario_dir = output_root / spec.name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nScenario {scenario_idx}/{len(scenarios)}: {spec.description}")
        for episode_idx in range(args.episodes):
            controllers = spec.controller_factory()
            seed = base_seed + (scenario_idx - 1) * args.episodes + episode_idx
            target_dir = _episode_output_dir(scenario_dir, episode_idx + 1, seed)
            target_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Episode {episode_idx + 1}/{args.episodes} | seed={seed}")
            start = time.time()
            state, decisions, returns, steps = run_episode(game, controllers, seed)
            duration = time.time() - start
            snapshot_path, animation_path, wintak_snapshot_path = _save_visuals(
                state,
                target_dir,
                make_snapshot=not args.no_snapshot,
                make_animation=not args.no_animation,
                export_wintak_snapshot=not args.no_wintak_snapshot,
                time_step=args.time_step,
                fps=args.fps,
            )
            summary_path = _write_summary(
                target_dir,
                seed,
                steps,
                returns,
                decisions,
                snapshot_path,
                animation_path,
                wintak_snapshot_path,
            )
            artifacts.append(
                EpisodeArtifacts(
                    scenario=spec.name,
                    seed=seed,
                    steps=steps,
                    returns=returns,
                    snapshot_path=snapshot_path,
                    animation_path=animation_path,
                    wintak_snapshot_path=wintak_snapshot_path,
                    summary_path=summary_path,
                )
            )
            print(
                "    returns → attacker {att:.2f} | defender {defn:.2f} | steps {steps} | elapsed {elapsed:.1f}s".format(
                    att=returns[0], defn=returns[1], steps=steps, elapsed=duration
                )
            )
            if snapshot_path:
                print(f"    snapshot: {snapshot_path}")
            if animation_path:
                print(f"    animation: {animation_path}")
            if wintak_snapshot_path:
                print(f"    WinTAK snapshot: {wintak_snapshot_path}")
            print(f"    episode summary: {summary_path}")

    print("\nFinished {count} scenario episode(s).".format(count=len(artifacts)))
    for artifact in artifacts:
        print(
            (
                "Scenario {scenario}: seed={seed} attacker={att:.1f} defender={defn:.1f} "
                "snapshot={snap} animation={anim} wintak_snapshot={wintak}"
            ).format(
                scenario=artifact.scenario,
                seed=artifact.seed,
                att=artifact.returns[0],
                defn=artifact.returns[1],
                snap=artifact.snapshot_path if artifact.snapshot_path else "-",
                anim=artifact.animation_path if artifact.animation_path else "-",
                wintak=artifact.wintak_snapshot_path if artifact.wintak_snapshot_path else "-",
            )
        )


if __name__ == "__main__":
    main()
