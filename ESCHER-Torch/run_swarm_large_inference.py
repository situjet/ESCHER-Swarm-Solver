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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pyspiel  # type: ignore
import torch

# Force a non-interactive Matplotlib backend before importing the demo helpers.
os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parent
SWARM_REPO_PATH = PROJECT_ROOT.parent / "Swarm-AD-Large-OpenSpiel-2"
if str(SWARM_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(SWARM_REPO_PATH))

from demo_animation import _build_animation as build_animation  # type: ignore
from demo_visualizer import render_snapshot  # type: ignore
from swarm_defense_large_game import SwarmDefenseLargeState  # type: ignore

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
    seed: int
    steps: int
    returns: Tuple[float, float]
    snapshot_path: Optional[Path]
    animation_path: Optional[Path]
    summary_path: Path


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


class PolicyActor:
    """Wraps a trained PolicyNetwork for action selection."""

    def __init__(self, policy_path: Path, device: str = "cpu") -> None:
        ensure_information_state_tensor()
        self.game = pyspiel.load_game(GAME_NAME)
        self.device = device
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

    def act(
        self,
        state: pyspiel.State,
        rng: random.Random,
        *,
        sampling: bool,
    ) -> Tuple[int, Dict[int, float]]:
        player = state.current_player()
        legal_actions = state.legal_actions(player)
        if not legal_actions:
            raise RuntimeError(f"No legal actions for player {player}")
        if len(legal_actions) == 1:
            return legal_actions[0], {legal_actions[0]: 1.0}
        info_state = np.asarray(state.information_state_tensor(player), dtype=np.float32)
        mask = _legal_mask(state, player, self.num_actions)
        info_tensor = torch.from_numpy(info_state).unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.network(info_tensor, mask_tensor)[0].cpu().numpy()
        action_probabilities = {action: float(max(probs[action], 0.0)) for action in legal_actions}
        if sampling:
            weights = [action_probabilities[action] for action in legal_actions]
            # Guard against extremely peaked or zeroed distributions.
            weights = [w if w > 0.0 else 1e-9 for w in weights]
            action = rng.choices(legal_actions, weights=weights, k=1)[0]
        else:
            action = max(legal_actions, key=lambda a: action_probabilities.get(a, 0.0))
        return action, action_probabilities


def _top_k(probabilities: Dict[int, float], k: int = 5) -> List[Tuple[int, float]]:
    return sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:k]


def run_episode(
    actor: PolicyActor,
    seed: int,
    *,
    sampling: bool,
) -> Tuple[SwarmDefenseLargeState, List[Dict[str, object]], Tuple[float, float], int]:
    rng = random.Random(seed)
    state = actor.game.new_initial_state()
    decision_trace: List[Dict[str, object]] = []
    steps = 0

    while not state.is_terminal():
        player = state.current_player()
        if player == pyspiel.PlayerId.CHANCE:
            action = _sample_chance_action(state, rng)
            decision_trace.append({"step": steps, "player": "chance", "action": action})
        else:
            action, probs = actor.act(state, rng, sampling=sampling)
            decision_trace.append(
                {
                    "step": steps,
                    "player": int(player),
                    "action": action,
                    "top_actions": _top_k(probs),
                }
            )
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
    time_step: float,
    fps: int,
) -> Tuple[Optional[Path], Optional[Path]]:
    snapshot_path = episode_dir / "snapshot.png" if make_snapshot else None
    animation_path = episode_dir / "animation.gif" if make_animation else None
    if snapshot_path is not None:
        render_snapshot(state, snapshot_path)
    if animation_path is not None:
        build_animation(state, animation_path, time_step=time_step, fps=fps)
    return snapshot_path, animation_path


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
        help="Number of evaluation episodes to run",
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
    print(f"Device: {args.device} | Episodes: {args.episodes} | Sampling: {args.sampling}")
    print(f"Outputs → {output_root}")

    actor = PolicyActor(policy_path, device=args.device)
    artifacts: List[EpisodeArtifacts] = []

    for episode_idx in range(args.episodes):
        seed = base_seed + episode_idx
        target_dir = _episode_output_dir(output_root, episode_idx + 1, seed)
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nEpisode {episode_idx + 1}/{args.episodes} | seed={seed}")
        start = time.time()
        state, decisions, returns, steps = run_episode(actor, seed, sampling=args.sampling)
        duration = time.time() - start
        snapshot_path, animation_path = _save_visuals(
            state,
            target_dir,
            make_snapshot=not args.no_snapshot,
            make_animation=not args.no_animation,
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
        )
        artifacts.append(
            EpisodeArtifacts(
                seed=seed,
                steps=steps,
                returns=returns,
                snapshot_path=snapshot_path,
                animation_path=animation_path,
                summary_path=summary_path,
            )
        )
        print(
            "  returns → attacker {att:.2f} | defender {defn:.2f} | steps {steps} | elapsed {elapsed:.1f}s".format(
                att=returns[0], defn=returns[1], steps=steps, elapsed=duration
            )
        )
        if snapshot_path:
            print(f"  snapshot: {snapshot_path}")
        if animation_path:
            print(f"  animation: {animation_path}")
        print(f"  episode summary: {summary_path}")

    print("\nFinished {count} episode(s).".format(count=len(artifacts)))
    for idx, artifact in enumerate(artifacts, start=1):
        print(
            "Episode {idx}: seed={seed} attacker={att:.1f} defender={defn:.1f} snapshot={snap} animation={anim}".format(
                idx=idx,
                seed=artifact.seed,
                att=artifact.returns[0],
                defn=artifact.returns[1],
                snap=artifact.snapshot_path if artifact.snapshot_path else "-",
                anim=artifact.animation_path if artifact.animation_path else "-",
            )
        )


if __name__ == "__main__":
    main()
