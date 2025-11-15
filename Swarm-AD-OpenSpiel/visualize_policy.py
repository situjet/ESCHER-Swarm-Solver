"""Visualize a trained ESCHER policy for the Swarm Defense game."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import pyspiel

# Add ESCHER-Torch to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "ESCHER-Torch"))

from ESCHER_Torch.eschersolver import PolicyNetwork
from swarm_defense_game import SwarmDefenseState
from demo_visualizer import (
    _sample_chance_action,
    _defender_policy,
    _attacker_policy,
    render_snapshot,
    _count_outcomes,
    OUTPUT_PATH,
)


class PolicyWrapper:
    """Wrapper to use trained policy network for action selection."""
    
    def __init__(
        self,
        policy_network: PolicyNetwork,
        player_id: int,
        device: str = "cpu",
    ):
        self.policy_network = policy_network
        self.player_id = player_id
        self.device = device
        self.policy_network.eval()
        self.policy_network.to(device)
    
    def get_action(self, state: SwarmDefenseState, rng: random.Random, use_sampling: bool = False) -> int:
        """Get action from the trained policy.
        
        Args:
            state: Current game state
            rng: Random number generator for sampling
            use_sampling: If True, sample from distribution; if False, take argmax
        """
        cur_player = state.current_player()
        if cur_player != self.player_id:
            raise ValueError(f"Policy is for player {self.player_id} but state is for player {cur_player}")
        
        legal_actions = state.legal_actions(cur_player)
        if len(legal_actions) == 1:
            return legal_actions[0]
        
        # Get info state and mask
        info_state = np.array(state.information_state_tensor(cur_player), dtype=np.float32)
        mask = np.array(state.legal_actions_mask(cur_player), dtype=np.float32)
        
        # Forward pass through network
        with torch.no_grad():
            info_tensor = torch.from_numpy(info_state).unsqueeze(0).to(self.device)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(self.device)
            probs = self.policy_network(info_tensor, mask_tensor)[0].cpu().numpy()
        
        # Sample or take argmax
        if use_sampling:
            # Sample according to probabilities
            action = rng.choices(legal_actions, weights=[probs[a] for a in legal_actions])[0]
        else:
            # Take the action with highest probability
            action = max(legal_actions, key=lambda a: probs[a])
        
        return action


def load_policy(
    policy_path: Path,
    game: pyspiel.Game,
    device: str = "cpu",
    policy_layers: Tuple[int, ...] = (256, 128),
) -> PolicyNetwork:
    """Load a trained policy network from a checkpoint.
    
    Args:
        policy_path: Path to the .pt file containing policy weights
        game: The OpenSpiel game (needed to get state tensor size)
        device: Device to load model on
        policy_layers: Architecture of the policy network (must match training)
    
    Returns:
        Loaded PolicyNetwork
    """
    # Get dimensions from game
    initial_state = game.new_initial_state()
    input_size = len(initial_state.information_state_tensor(0))
    num_actions = game.num_distinct_actions()
    
    # Create network with same architecture as training
    policy_net = PolicyNetwork(
        input_size=input_size,
        hidden_layers=policy_layers,
        num_actions=num_actions,
    )
    
    # Load weights
    state_dict = torch.load(str(policy_path), map_location=device)
    policy_net.load_state_dict(state_dict)
    policy_net.eval()
    
    return policy_net


def play_episode_with_policy(
    policy_path_p0: Optional[Path] = None,
    policy_path_p1: Optional[Path] = None,
    seed: Optional[int] = None,
    use_sampling: bool = False,
    device: str = "cpu",
) -> Tuple[SwarmDefenseState, int]:
    """Play an episode with trained policies for one or both players.
    
    Args:
        policy_path_p0: Path to trained policy .pt file for player 0 (attacker), None = use default
        policy_path_p1: Path to trained policy .pt file for player 1 (defender), None = use default
        seed: Random seed for reproducibility
        use_sampling: If True, sample from policy; if False, take argmax
        device: Device for policy network inference
    
    Returns:
        Final state and seed used
    """
    if seed is None:
        seed = random.SystemRandom().randint(0, 2**31 - 1)
    rng = random.Random(seed)
    
    import swarm_defense_game  # noqa: F401  # Registers the custom game
    
    game = pyspiel.load_game("swarm_defense")
    state = game.new_initial_state()
    
    # Load policies for both players if provided
    policy_wrapper_p0 = None
    policy_wrapper_p1 = None
    
    if policy_path_p0 is not None:
        print(f"Loading Player 0 (Attacker) policy from: {policy_path_p0}")
        policy_net_p0 = load_policy(policy_path_p0, game, device=device)
        policy_wrapper_p0 = PolicyWrapper(policy_net_p0, 0, device=device)
        print(f"✓ Player 0 policy loaded")
    
    if policy_path_p1 is not None:
        print(f"Loading Player 1 (Defender) policy from: {policy_path_p1}")
        policy_net_p1 = load_policy(policy_path_p1, game, device=device)
        policy_wrapper_p1 = PolicyWrapper(policy_net_p1, 1, device=device)
        print(f"✓ Player 1 policy loaded")
    
    # Play episode
    step = 0
    while not state.is_terminal():
        player = state.current_player()
        
        if player == pyspiel.PlayerId.CHANCE:
            action = _sample_chance_action(state, rng)
        elif player == 0:
            if policy_wrapper_p0 is not None:
                # Use trained policy for attacker
                action = policy_wrapper_p0.get_action(state, rng, use_sampling=use_sampling)
            else:
                # Use default attacker policy
                action = _attacker_policy(state, rng)
        else:  # player == 1
            if policy_wrapper_p1 is not None:
                # Use trained policy for defender
                action = policy_wrapper_p1.get_action(state, rng, use_sampling=use_sampling)
            else:
                # Use default defender policy
                action = _defender_policy(state, rng)
        
        state.apply_action(action)
        step += 1
    
    return state, seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a trained ESCHER policy for Swarm Defense"
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Path to policy.pt file (for both players, or use --policy-p0/--policy-p1 for separate policies)",
    )
    parser.add_argument(
        "--policy-p0",
        type=str,
        default=None,
        help="Path to policy.pt file for Player 0 (Attacker)",
    )
    parser.add_argument(
        "--policy-p1",
        type=str,
        default=None,
        help="Path to policy.pt file for Player 1 (Defender)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--sampling",
        action="store_true",
        help="Sample from policy distribution instead of taking argmax",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for inference (cpu or cuda)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for visualization (default: Visualizer/swarm_defense_demo.png)",
    )
    
    args = parser.parse_args()
    
    # Handle policy arguments
    policy_path_p0 = None
    policy_path_p1 = None
    
    if args.policy:
        # Use same policy for both players
        policy_path_p0 = Path(args.policy)
        policy_path_p1 = Path(args.policy)
        if not policy_path_p0.exists():
            print(f"Error: Policy file not found: {policy_path_p0}")
            sys.exit(1)
    else:
        # Use separate policies
        if args.policy_p0:
            policy_path_p0 = Path(args.policy_p0)
            if not policy_path_p0.exists():
                print(f"Error: Player 0 policy file not found: {policy_path_p0}")
                sys.exit(1)
        if args.policy_p1:
            policy_path_p1 = Path(args.policy_p1)
            if not policy_path_p1.exists():
                print(f"Error: Player 1 policy file not found: {policy_path_p1}")
                sys.exit(1)
    
    if policy_path_p0 is None and policy_path_p1 is None:
        print("Error: At least one policy must be specified (use --policy, --policy-p0, or --policy-p1)")
        sys.exit(1)
    
    output_path = Path(args.output) if args.output else OUTPUT_PATH
    
    # Play episode
    print("\n" + "="*60)
    print("Playing Swarm Defense Episode")
    print("="*60)
    print(f"Player 0 (Attacker): {'Trained policy' if policy_path_p0 else 'Default heuristic'}")
    print(f"Player 1 (Defender): {'Trained policy' if policy_path_p1 else 'Default heuristic'}")
    print(f"Sampling: {'Yes' if args.sampling else 'No (argmax)'}")
    print(f"Device: {args.device}")
    print()
    
    state, seed = play_episode_with_policy(
        policy_path_p0=policy_path_p0,
        policy_path_p1=policy_path_p1,
        seed=args.seed,
        use_sampling=args.sampling,
        device=args.device,
    )
    
    # Render visualization
    print("\nGenerating visualization...")
    render_snapshot(state, output_path)
    
    # Print results
    returns = state.returns()
    snapshot = state.snapshot()
    ad_kills, interceptor_kills, survivors, ad_attrit = _count_outcomes(snapshot["drones"])
    total_drones = len(snapshot["drones"])
    
    print("\n" + "="*60)
    print("Episode Results")
    print("="*60)
    print(f"Seed used: {seed}")
    print(f"Attacker damage: {returns[0]:.1f}")
    print(f"Defender reward: {returns[1]:.1f}")
    print(f"\nTotal attacking drones: {total_drones}")
    print(f"\nDrone Outcomes:")
    print(f"  - AD kills (intercepts): {ad_kills}")
    print(f"  - AD-target strikes: {ad_attrit}")
    print(f"  - Interceptor kills: {interceptor_kills}")
    print(f"  - Survivors: {survivors}")
    print(f"\n  Total: {ad_kills + ad_attrit + interceptor_kills + survivors} of {total_drones} drones")
    print(f"\nVisualization saved to: {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()

