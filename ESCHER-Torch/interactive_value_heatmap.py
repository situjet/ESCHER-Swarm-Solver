#!/usr/bin/env python3
"""
Interactive script to set target positions and visualize value function heatmap
for different AD unit placements.

Usage:
    python interactive_value_heatmap.py [checkpoint_dir]
    
    If checkpoint_dir is not provided, uses default path.
    
    The script will:
    1. Load the game and trained networks from checkpoint
    2. Prompt you to set target positions interactively
    3. Optionally set first AD position
    4. Evaluate value function for all possible second AD positions
    5. Display and save a heatmap visualization
    
Example:
    python interactive_value_heatmap.py
    # Then enter target positions like: 5,2 for row 5, column 2
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import pyspiel
import importlib
import importlib.util
from itertools import permutations

# Add ESCHER-Torch to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from ESCHER_Torch.eschersolver import PolicyNetwork, ValueNetwork


def load_game_and_networks(checkpoint_dir: Path):
    """Load game and networks from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_dir}")
    
    # Load metadata
    metadata_path = checkpoint_dir / "metadata.pt"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")
    metadata = torch.load(metadata_path, map_location="cpu")
    
    # Load game
    repo_root = checkpoint_dir.resolve().parents[3]
    reset_game_path = repo_root / 'Swarm-AD-OpenSpielReset' / 'swarm_defense_game.py'
    
    if not reset_game_path.exists():
        raise FileNotFoundError(f"Game file not found at {reset_game_path}")
    
    # Load game module
    module_name = 'swarm_defense_game'
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
        module = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(module_name, reset_game_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    
    # Load game with parameters
    game_params = metadata.get('game_config', {}).get('parameters', {}) or {}
    if game_params:
        param_parts = [f"{key}={value}" for key, value in game_params.items()]
        spec_string = f"swarm_defense({','.join(param_parts)})"
        game = pyspiel.load_game(spec_string)
    else:
        game = pyspiel.load_game('swarm_defense')
    
    # Get dimensions
    initial_state = game.new_initial_state()
    state_tensor_size = len(initial_state.information_state_tensor(0))
    num_actions = game.num_distinct_actions()
    
    # Load networks
    policy_net = PolicyNetwork(state_tensor_size, (256, 128), num_actions)
    policy_path = checkpoint_dir / "policy.pt"
    if policy_path.exists():
        policy_net.load_state_dict(torch.load(policy_path, map_location="cpu"))
    policy_net.eval()
    
    # Load value network
    value_path = checkpoint_dir / "value.pt"
    if not value_path.exists():
        raise FileNotFoundError(f"Value network not found at {value_path}")
    
    value_checkpoint = torch.load(value_path, map_location="cpu")
    value_input_size = value_checkpoint['net.0.weight'].shape[1]
    value_net = ValueNetwork(value_input_size, (256, 128))
    value_net.load_state_dict(value_checkpoint)
    value_net.eval()
    
    print(f"✓ Game loaded: {game.get_type().short_name}")
    print(f"✓ Policy network loaded: {state_tensor_size} -> (256, 128) -> {num_actions}")
    print(f"✓ Value network loaded: {value_input_size} -> (256, 128) -> 1")
    
    return game, policy_net, value_net


def set_target_position(state, target_pos: tuple):
    """Set a target at the specified position (row, col)."""
    if state.current_player() != pyspiel.PlayerId.CHANCE:
        raise ValueError("Not at a chance node for target placement")
    
    chance_outcomes = state.chance_outcomes()
    legal_actions = [a for (a, _) in chance_outcomes]
    
    # Find action that places target at desired position
    for action in legal_actions:
        try:
            decoded_pos = state._decode_target_position_action(action)
            if decoded_pos == target_pos:
                state.apply_action(action)
                return action
        except:
            continue
    
    # If not found, use first legal action and warn
    action = legal_actions[0]
    decoded_pos = state._decode_target_position_action(action)
    print(f"  Warning: Position {target_pos} not available, using {decoded_pos} instead")
    state.apply_action(action)
    return action


def set_target_values(state, desired_values=None):
    """Set target values to desired assignment.
    
    Args:
        state: Game state at TARGET_VALUES phase
        desired_values: List of values to assign to targets in order, e.g., [10.0, 20.0]
                       If None, uses first permutation
    """
    if state.current_player() != pyspiel.PlayerId.CHANCE:
        raise ValueError("Not at a chance node for target values")
    
    chance_outcomes = state.chance_outcomes()
    if len(chance_outcomes) == 0:
        return None
    
    if desired_values is None:
        # Use first permutation
        action = chance_outcomes[0][0]
        state.apply_action(action)
        return action
    
    # Find permutation that matches desired values
    value_options = state.config.get_target_value_options()
    num_targets = state.config.num_targets
    
    # Check if desired values match available options
    if len(desired_values) != num_targets:
        raise ValueError(f"Number of desired values ({len(desired_values)}) doesn't match number of targets ({num_targets})")
    
    # Find which indices in value_options correspond to desired values
    try:
        value_indices = [value_options.index(val) for val in desired_values]
    except ValueError as e:
        available = value_options[:num_targets]
        raise ValueError(f"Desired values {desired_values} not all in available options {available}")
    
    # Find permutation that matches
    perms = list(permutations(range(num_targets)))
    
    matching_perm = None
    for perm in perms:
        if [value_options[perm[i]] for i in range(num_targets)] == desired_values:
            matching_perm = perm
            break
    
    if matching_perm is None:
        # Try to find closest match or use first permutation
        print(f"  Warning: Could not find exact permutation for {desired_values}, using first available")
        action = chance_outcomes[0][0]
        state.apply_action(action)
        return action
    
    # Find action corresponding to this permutation
    perm_idx = perms.index(matching_perm)
    target_value_base = state.action_info["target_value_base"]
    action = target_value_base + perm_idx
    
    # Verify this action is legal
    legal_actions = [a for (a, _) in chance_outcomes]
    if action in legal_actions:
        state.apply_action(action)
        return action
    else:
        print(f"  Warning: Action {action} for permutation {matching_perm} not legal, using first available")
        action = chance_outcomes[0][0]
        state.apply_action(action)
        return action


def evaluate_value(state, value_net):
    """Evaluate value function for current state."""
    # Value network takes concatenated info states from both players
    info_state_0 = state.information_state_tensor(0)
    info_state_1 = state.information_state_tensor(1)
    hist_state = np.append(info_state_0, info_state_1).astype(np.float32)
    
    with torch.no_grad():
        hist_tensor = torch.FloatTensor(hist_state).unsqueeze(0)
        # Value network doesn't use mask
        mask = torch.ones(1, 1)  # Dummy mask
        value = value_net(hist_tensor, mask).item()
    
    return value


def create_value_heatmap(game, policy_net, value_net, target_positions, first_ad_position=None, output_dir=None):
    """Create heatmap of value function for different second AD positions."""
    print(f"\n=== Creating Value Function Heatmap ===")
    print(f"Target positions: {target_positions}")
    if first_ad_position:
        print(f"First AD position: {first_ad_position}")
    
    # Create base state with targets
    state_base = game.new_initial_state()
    
    # Set target positions
    print("\nSetting target positions...")
    for i, target_pos in enumerate(target_positions):
        set_target_position(state_base, target_pos)
        if hasattr(state_base, '_target_positions'):
            print(f"  Target {i+1}: {state_base._target_positions[-1]}")
    
    # Set target values (10 and 20 for targets 1 and 2)
    print("\nSetting target values...")
    desired_target_values = [10.0, 20.0]  # Target 1 gets 10, Target 2 gets 20
    set_target_values(state_base, desired_values=desired_target_values)
    print(f"  Target values set to: {desired_target_values}")
    
    # Place first AD if specified
    if first_ad_position:
        print(f"\nPlacing first AD at {first_ad_position}...")
        if state_base.current_player() == 1 and state_base.phase().name == "AD_PLACEMENT":
            legal_actions = state_base.legal_actions(1)
            # Find action for first AD position
            for action in legal_actions:
                try:
                    decoded_pos = state_base._decode_ad_position(action)
                    if decoded_pos == first_ad_position:
                        state_base.apply_action(action)
                        print(f"  First AD placed at {first_ad_position}")
                        break
                except:
                    continue
            else:
                # Use first legal action if position not found
                action = legal_actions[0]
                decoded_pos = state_base._decode_ad_position(action)
                state_base.apply_action(action)
                print(f"  Warning: {first_ad_position} not available, using {decoded_pos}")
    
    # Check if ready for second AD placement
    if state_base.current_player() != 1 or state_base.phase().name != "AD_PLACEMENT":
        print(f"  Not ready for AD placement (phase: {state_base.phase().name}, player: {state_base.current_player()})")
        return None
    
    legal_ad_actions = state_base.legal_actions(1)
    print(f"\nEvaluating {len(legal_ad_actions)} second AD positions...")
    
    # Evaluate each position
    results = []
    for idx, ad_action in enumerate(legal_ad_actions):
        if idx % 10 == 0 and idx > 0:
            print(f"  Progress: {idx}/{len(legal_ad_actions)}")
        
        # Clone state and place second AD
        state = state_base.clone()
        state.apply_action(ad_action)
        
        # Decode position
        try:
            row, col = state._decode_ad_position(ad_action)
        except:
            if hasattr(state, '_ad_units') and len(state._ad_units) >= 2:
                row, col = state._ad_units[1].row, state._ad_units[1].col
            else:
                row, col = -1, -1
        
        # Evaluate value function
        value = evaluate_value(state, value_net)
        
        results.append({
            'row': row,
            'col': col,
            'value': value,
            'action': ad_action
        })
    
    print(f"✓ Evaluated {len(results)} positions")
    
    # Create heatmap
    if len(results) > 0:
        rows = [r['row'] for r in results]
        cols = [r['col'] for r in results]
        values = [r['value'] for r in results]
        
        # Get target positions for visualization
        if hasattr(state_base, '_targets') and len(state_base._targets) > 0:
            target_positions_viz = [(t.row, t.col, t.value) for t in state_base._targets]
        elif hasattr(state_base, '_target_positions') and len(state_base._target_positions) > 0:
            target_positions_viz = [(pos[0], pos[1], None) for pos in state_base._target_positions]
        else:
            target_positions_viz = target_positions
        
        # Get first AD position
        if hasattr(state_base, '_ad_units') and len(state_base._ad_units) > 0:
            first_ad_viz = (state_base._ad_units[0].row, state_base._ad_units[0].col)
        else:
            first_ad_viz = first_ad_position
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Scatter plot with value as color
        scatter = ax.scatter(cols, rows, c=values, cmap='viridis', s=200, 
                            alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Plot targets
        if target_positions_viz:
            target_rows = [t[0] for t in target_positions_viz]
            target_cols = [t[1] for t in target_positions_viz]
            ax.scatter(target_cols, target_rows, c='red', marker='*', s=600, 
                      alpha=0.9, edgecolors='darkred', linewidth=2, 
                      label='Targets', zorder=5)
            for i, (row, col, value) in enumerate(target_positions_viz):
                ax.annotate(f'T{i+1}', (col, row), xytext=(5, 5), 
                           textcoords='offset points', fontsize=12, 
                           fontweight='bold', color='red')
        
        # Plot first AD
        if first_ad_viz:
            ax.scatter([first_ad_viz[1]], [first_ad_viz[0]], c='orange', 
                      marker='s', s=400, alpha=0.9, edgecolors='darkorange', 
                      linewidth=2, label='First AD', zorder=5)
            ax.annotate('AD1', (first_ad_viz[1], first_ad_viz[0]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=12, fontweight='bold', color='darkorange')
        
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)
        ax.set_title('Value Function Heatmap for Second AD Placement\n' +
                    f'Targets: {target_positions}, First AD: {first_ad_viz}', 
                    fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
        ax.legend(loc='upper right', fontsize=10)
        plt.colorbar(scatter, ax=ax, label='Value Function')
        
        plt.tight_layout()
        
        # Save and show
        if output_dir is None:
            output_path = Path("value_heatmap.png")
        else:
            output_path = output_dir / "value_heatmap.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved heatmap to: {output_path}")
        plt.show()
        
        # Print top positions
        results_sorted = sorted(results, key=lambda x: x['value'], reverse=True)
        print(f"\n=== Top 10 Second AD Positions (by Value) ===")
        for i, result in enumerate(results_sorted[:10], 1):
            print(f"{i}. Position ({result['row']:2d}, {result['col']:2d}): "
                  f"value={result['value']:.4f}")
        
        return results
    
    return None


def main():
    """Main interactive function."""
    global checkpoint_dir
    
    # Default checkpoint path (modify as needed)
    checkpoint_dir = Path("/carnegie/nobackup/users/lvulpius/ESCHER-Swarm-Solver/ESCHER-Torch/results/swarm_defense/2025_11_16_04_12_45/")
    
    if len(sys.argv) > 1:
        checkpoint_dir = Path(sys.argv[1])
    
    # Load game and networks
    game, policy_net, value_net = load_game_and_networks(checkpoint_dir)
    
    # Get valid target cells
    target_cells = game.config.get_target_candidate_cells()
    print(f"\nValid target positions (rows {game.config.bottom_half_start + 1} to {game.config.grid_rows - 1}, "
          f"cols 1 to {game.config.grid_cols - 2}):")
    print(f"  {target_cells[:10]}..." if len(target_cells) > 10 else f"  {target_cells}")
    
    # Interactive target setting
    print(f"\n=== Interactive Target Setting ===")
    print(f"Enter target positions as 'row,col' (e.g., '5,2' for row 5, column 2)")
    print(f"Press Enter with empty input to use defaults or finish")
    
    target_positions = []
    num_targets = game.config.num_targets
    
    for i in range(num_targets):
        user_input = input(f"\nTarget {i+1} position (row,col) or Enter for default: ").strip()
        if user_input:
            try:
                row, col = map(int, user_input.split(','))
                target_pos = (row, col)
                if target_pos in target_cells:
                    target_positions.append(target_pos)
                    print(f"  ✓ Set target {i+1} to {target_pos}")
                else:
                    print(f"  ✗ {target_pos} is not a valid target position")
                    print(f"  Using first available position instead")
                    target_positions.append(target_cells[0])
            except:
                print(f"  ✗ Invalid input, using first available position")
                target_positions.append(target_cells[0])
        else:
            # Use first available positions
            if i < len(target_cells):
                target_positions.append(target_cells[i])
            else:
                target_positions.append(target_cells[0])
            print(f"  Using default: {target_positions[-1]}")
    
    # Optional: Set first AD position
    first_ad_position = None
    user_input = input(f"\nFirst AD position (row,col) or Enter to skip: ").strip()
    if user_input:
        try:
            row, col = map(int, user_input.split(','))
            first_ad_position = (row, col)
            print(f"  ✓ Set first AD to {first_ad_position}")
        except:
            print(f"  ✗ Invalid input, will use first available AD position")
    
    # Create heatmap
    results = create_value_heatmap(game, policy_net, value_net, target_positions, first_ad_position, checkpoint_dir)
    
    if results:
        print(f"\n✓ Analysis complete!")
    else:
        print(f"\n✗ Failed to create heatmap")


if __name__ == "__main__":
    main()

