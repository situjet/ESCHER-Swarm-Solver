"""Regret-matching inference for Swarm Defense Large v2.

This helper mirrors run_swarm_large_inference.py but chooses actions via
regret matching: the learned regret networks produce positive regrets that
are normalized into a behavior policy, optionally blended with the trained
policy network for stability.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pyspiel  # type: ignore
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
SWARM_REPO_PATH = PROJECT_ROOT.parent / "Swarm-AD-Large-OpenSpiel-2"
if str(SWARM_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(SWARM_REPO_PATH))

from ESCHER_Torch.eschersolver import PolicyNetwork, RegretNetwork
from run_escher_torch_swarm_2 import (  # type: ignore
    GAME_NAME,
    NETWORK_LAYERS,
    RESULT_DIR_NAME,
    ensure_information_state_tensor,
)
from run_swarm_large_inference import (  # type: ignore
    EpisodeArtifacts,
    ScenarioSpec,
    Controller,
    NaiveAttackerController,
    NaiveDefenderController,
    run_episode,
    _episode_output_dir,
    _save_visuals,
    _write_summary,
    _resolve_checkpoint_dir,
    _legal_mask,
    _top_k,
)
from swarm_defense_large_game import INTERCEPT_END_ACTION, Phase  # type: ignore

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / RESULT_DIR_NAME / "regret_matching_runs"


class RegretMatchingController(Controller):
    """Action selection via regret matching with optional policy blending."""

    def __init__(
        self,
        game: pyspiel.Game,
        checkpoint_dir: Path,
        player_id: int,
        *,
        device: str = "cpu",
        sampling: bool = False,
        policy_mix: float = 0.15,
        min_regret_mass: float = 1e-6,
    ) -> None:
        self.game = game
        self.player_id = player_id
        self.device = torch.device(device)
        self.sampling = sampling
        self.policy_mix = float(max(0.0, min(1.0, policy_mix)))
        self.min_regret_mass = max(1e-9, float(min_regret_mass))
        template_state = self.game.new_initial_state()
        self.info_dim = len(template_state.information_state_tensor(0))
        self.num_actions = self.game.num_distinct_actions()

        policy_path = checkpoint_dir / "policy.pt"
        if not policy_path.exists():
            raise FileNotFoundError(f"policy.pt not found in {checkpoint_dir}")
        regret_path = checkpoint_dir / f"regret_player{player_id}.pt"
        if not regret_path.exists():
            raise FileNotFoundError(f"regret_player{player_id}.pt not found in {checkpoint_dir}")

        self.policy_network = PolicyNetwork(
            input_size=self.info_dim,
            hidden_layers=NETWORK_LAYERS,
            num_actions=self.num_actions,
        ).to(self.device)
        policy_state = torch.load(str(policy_path), map_location=self.device)
        self.policy_network.load_state_dict(policy_state)
        self.policy_network.eval()

        self.regret_network = RegretNetwork(
            input_size=self.info_dim,
            hidden_layers=NETWORK_LAYERS,
            num_actions=self.num_actions,
        ).to(self.device)
        regret_state = torch.load(str(regret_path), map_location=self.device)
        self.regret_network.load_state_dict(regret_state)
        self.regret_network.eval()

    def _distribution_from_networks(
        self,
        info_state: np.ndarray,
        mask: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        info_tensor = torch.from_numpy(info_state).unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        with torch.no_grad():
            regret_values = torch.clamp(self.regret_network(info_tensor, mask_tensor), min=0.0)[0]
            policy_probs = self.policy_network(info_tensor, mask_tensor)[0]
        regret_mass = float(regret_values.sum().item())
        if regret_mass <= self.min_regret_mass:
            combined = policy_probs.clone()
            source = "policy"
        else:
            combined = regret_values / max(regret_mass, 1e-8)
            source = "regret"
        if self.policy_mix > 0.0:
            combined = (1.0 - self.policy_mix) * combined + self.policy_mix * policy_probs
        mask_values = torch.from_numpy(mask).to(self.device)
        combined = combined * mask_values
        total = float(combined.sum().item())
        if total <= 0.0:
            legal_indices = np.where(mask > 0)[0]
            combined = torch.zeros_like(combined)
            if len(legal_indices) > 0:
                combined[legal_indices] = 1.0 / len(legal_indices)
            else:
                combined.fill_(1.0 / combined.numel())
            total = float(combined.sum().item())
            source = "uniform"
        probs = combined / total
        return {
            "probabilities": probs.cpu().numpy(),
            "policy_probs": policy_probs.cpu().numpy(),
            "regret_mass": regret_mass,
            "source": source,
        }

    def choose_action(self, state: pyspiel.State, rng: random.Random):
        player = state.current_player()
        if player != self.player_id:
            raise RuntimeError(f"Regret controller invoked for player {player}, expected {self.player_id}")
        legal_actions = state.legal_actions(player)
        if not legal_actions:
            raise RuntimeError(f"Player {player} has no legal actions")
        if len(legal_actions) == 1:
            action = legal_actions[0]
            return action, {"controller": "regret_matching"}
        info_state = np.asarray(state.information_state_tensor(player), dtype=np.float32)
        mask = _legal_mask(state, player, self.num_actions).astype(np.float32, copy=False)
        dist = self._distribution_from_networks(info_state, mask)
        probs = dist["probabilities"].copy()
        intercept_pass_scaling = None
        phase = state.phase() if hasattr(state, "phase") else None
        if (
            self.player_id == 1
            and phase == Phase.INTERCEPT_ASSIGNMENT
            and INTERCEPT_END_ACTION in legal_actions
        ):
            pass_prob = float(max(probs[INTERCEPT_END_ACTION], 0.0))
            if pass_prob > 0.0:
                scaled_pass = pass_prob * 0.25
                redistribution = pass_prob - scaled_pass
                other_actions = [a for a in legal_actions if a != INTERCEPT_END_ACTION]
                if other_actions and redistribution > 0.0:
                    other_mass = float(sum(max(probs[a], 0.0) for a in other_actions))
                    if other_mass <= 0.0:
                        increment = redistribution / len(other_actions)
                        for action_id in other_actions:
                            probs[action_id] += increment
                    else:
                        for action_id in other_actions:
                            share = max(probs[action_id], 0.0) / other_mass
                            probs[action_id] += redistribution * share
                    probs[INTERCEPT_END_ACTION] = scaled_pass
                    intercept_pass_scaling = {
                        "original": pass_prob,
                        "scaled": scaled_pass,
                    }
        probs = np.clip(probs, 0.0, None)
        total_mass = float(probs.sum())
        if total_mass > 0.0:
            probs /= total_mass
        action_probabilities = {action: float(max(probs[action], 0.0)) for action in legal_actions}
        if self.sampling:
            weights = [action_probabilities[a] if action_probabilities[a] > 0.0 else 1e-9 for a in legal_actions]
            action = rng.choices(legal_actions, weights=weights, k=1)[0]
        else:
            action = max(legal_actions, key=lambda a: action_probabilities.get(a, 0.0))
        metadata: Dict[str, object] = {
            "controller": "regret_matching",
            "probabilities": action_probabilities,
            "regret_mass": dist["regret_mass"],
            "prob_source": dist["source"],
            "policy_mix": self.policy_mix,
            "top_actions": _top_k(action_probabilities),
        }
        if intercept_pass_scaling is not None:
            metadata["intercept_pass_scaled"] = intercept_pass_scaling
        return action, metadata


def _build_scenarios(
    mode: str,
    attacker_controller: RegretMatchingController,
    defender_controller: RegretMatchingController,
) -> List[ScenarioSpec]:
    if mode == "single":
        return [
            ScenarioSpec(
                name="regret_vs_regret",
                description="Both players use regret-matching controllers",
                controller_factory=lambda: {0: attacker_controller, 1: defender_controller},
            )
        ]
    return [
        ScenarioSpec(
            name="defender_regret_vs_naive_attacker",
            description="Defender uses regret matching vs naive 10-drones attacker",
            controller_factory=lambda: {0: NaiveAttackerController(), 1: defender_controller},
        ),
        ScenarioSpec(
            name="attacker_regret_vs_naive_defender",
            description="Attacker regret matching vs naive 10-intercepts defender",
            controller_factory=lambda: {0: attacker_controller, 1: NaiveDefenderController()},
        ),
        ScenarioSpec(
            name="regret_vs_regret",
            description="Both players use regret-matching controllers",
            controller_factory=lambda: {0: attacker_controller, 1: defender_controller},
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with regret-matching controllers for Swarm Defense Large v2",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint directory containing policy.pt and regret_player*.pt",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for inference",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Episodes per scenario",
    )
    parser.add_argument(
        "--scenario-mode",
        type=str,
        choices=["triad", "single"],
        default="triad",
        help="single=regret vs regret only, triad=regret + naive matchups",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base RNG seed (auto if omitted)",
    )
    parser.add_argument(
        "--sampling",
        action="store_true",
        help="Sample actions instead of greedy argmax",
    )
    parser.add_argument(
        "--policy-mix",
        type=float,
        default=0.15,
        help="Blend factor with the policy network (0=pure regret, 1=pure policy)",
    )
    parser.add_argument(
        "--min-regret-mass",
        type=float,
        default=1e-6,
        help="Fallback threshold: below this, revert to policy distribution",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for snapshots/animations/summaries",
    )
    parser.add_argument("--no-snapshot", action="store_true", help="Skip PNG snapshot export")
    parser.add_argument("--no-animation", action="store_true", help="Skip GIF animation export")
    parser.add_argument(
        "--no-wintak-snapshot",
        action="store_true",
        help="Skip WinTAK/CoT JSON snapshot export",
    )
    parser.add_argument("--time-step", type=float, default=0.25, help="Animation timestep (seconds)")
    parser.add_argument("--fps", type=int, default=12, help="Animation frames per second")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_dir = _resolve_checkpoint_dir(args.checkpoint)
    ensure_information_state_tensor()
    game = pyspiel.load_game(GAME_NAME)
    attacker_controller = RegretMatchingController(
        game,
        checkpoint_dir,
        player_id=0,
        device=args.device,
        sampling=args.sampling,
        policy_mix=args.policy_mix,
        min_regret_mass=args.min_regret_mass,
    )
    defender_controller = RegretMatchingController(
        game,
        checkpoint_dir,
        player_id=1,
        device=args.device,
        sampling=args.sampling,
        policy_mix=args.policy_mix,
        min_regret_mass=args.min_regret_mass,
    )
    scenarios = _build_scenarios(args.scenario_mode, attacker_controller, defender_controller)

    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    base_seed = args.seed if args.seed is not None else random.SystemRandom().randint(0, 2**31 - 1)

    print("Swarm Defense Large v2 | Regret-Matching Inference")
    print(f"Checkpoint: {checkpoint_dir}")
    print(
        "Device: {device} | Scenario mode: {mode} | Episodes/scenario: {episodes} | Sampling: {sampling} | Policy mix: {mix}".format(
            device=args.device,
            mode=args.scenario_mode,
            episodes=args.episodes,
            sampling=args.sampling,
            mix=args.policy_mix,
        )
    )
    print(f"Outputs → {output_root}")

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
            start_time = time.time()
            state, decisions, returns, steps = run_episode(game, controllers, seed)
            duration = time.time() - start_time
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
                    att=returns[0],
                    defn=returns[1],
                    steps=steps,
                    elapsed=duration,
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
