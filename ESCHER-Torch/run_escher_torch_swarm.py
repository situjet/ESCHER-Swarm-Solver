"""
ESCHER-Torch training for Swarm Defense game.

Performance Notes:
- Custom __deepcopy__ optimization: ~1-2 trees/second
- 100 traversals per player = ~100-200 seconds per iteration
- 20 iterations = ~30-60 minutes total training time
- GPU used for neural network training, CPU for tree traversal

Settings (balanced for convergence vs. time):
- 20 iterations, 100 traversals/player, 20 value traversals
- Batch sizes: 256 regret, 128 value, 256 policy
- Training steps: 200 per network

IMPORTANT: Batch size must be < samples collected!
- 100 traversals → ~200-400 samples → max batch size ~256
- 1000 traversals → ~2000-4000 samples → max batch size ~2000

Parallelization:
- Set num_workers > 1 to use multiple CPU cores for tree traversal
- Parallel traversal: 4 workers → ~2-3x speedup (diminishing returns beyond 4)
- Networks automatically moved to CPU for workers, then back to GPU for training
- Each worker traverses trees independently and samples are merged

For faster testing: reduce num_iterations and num_traversals
For better convergence: increase to 50+ iterations, 500+ traversals (and scale batch sizes)
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pyspiel
import torch

try:
    import yaml
except ImportError:  # pragma: no cover - handled at runtime if config is requested
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT.parent / "Swarm-AD-OpenSpielReset"))
sys.path.insert(0, str(PROJECT_ROOT.parent / "Swarm-AD-OpenSpiel"))

import swarm_defense_game  # noqa: F401
from ESCHER_Torch import ESCHERSolverTorch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ESCHER-Torch on the Swarm Defense game.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a YAML configuration file containing game/training parameters.",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Force test-mode hyperparameters regardless of YAML configuration.",
    )
    return parser.parse_args()


def _load_yaml_config(config_path: Optional[Path]) -> Dict[str, Any]:
    if config_path is None:
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to load configuration files (pip install pyyaml).")
    expanded = config_path.expanduser()
    with expanded.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file {expanded} must contain a top-level mapping.")
    return data


def main() -> None:
    args = _parse_args()
    yaml_config = _load_yaml_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_cpu_threads = int(yaml_config.get("num_cpu_threads", 1))
    
    training_overrides_raw = yaml_config.get("training", {})
    if training_overrides_raw and not isinstance(training_overrides_raw, dict):
        raise ValueError("`training` section in YAML must be a mapping.")
    training_overrides = dict(training_overrides_raw or {})
    
    yaml_test_mode = training_overrides.pop("test_mode", None)
    if yaml_test_mode is None:
        yaml_test_mode = yaml_config.get("test_mode")
    test_mode = bool(yaml_test_mode) if yaml_test_mode is not None else False
    if args.test_mode:
        test_mode = True
    
    test_defaults = {
        "num_iterations": 3,
        "num_traversals": 10,
        "num_val_traversals": 5,
        "num_workers": 1,
        "batch_size_regret": 60,
        "batch_size_value": 60,
        "batch_size_policy": 60,
        "train_steps": 50,
    }
    prod_defaults = {
        "num_iterations": 8,
        "num_traversals": 100,
        "num_val_traversals": 10,
        "num_workers": 1,
        "batch_size_regret": 50,
        "batch_size_value": 40,
        "batch_size_policy": 64,
        "train_steps": 200,
    }
    training_config = (test_defaults if test_mode else prod_defaults).copy()
    training_config.update(training_overrides)
    
    game_params_section = (
        yaml_config.get("game")
        or yaml_config.get("game_parameters")
        or yaml_config.get("game_params")
    )
    if game_params_section is not None and not isinstance(game_params_section, dict):
        raise ValueError("`game` configuration must be provided as a mapping.")
    game_params = dict(game_params_section or {})
    
    # Set PyTorch CPU threads early (before any torch operations)
    # This controls how many CPU threads PyTorch uses inside matrix/vector ops.
    # It is separate from ESCHER's tree-traversal workers.
    torch.set_num_threads(num_cpu_threads)
    torch.set_num_interop_threads(num_cpu_threads)
    print(f"🔧 PyTorch CPU threads: {num_cpu_threads} (affects tensor math, not traversal workers)")
    
    if game_params:
        print("📄 Loading swarm_defense with YAML parameters:")
        for key, value in game_params.items():
            print(f"   - {key}: {value}")
        print()
    game = pyspiel.load_game("swarm_defense", params=game_params)
        
    initial_state = game.new_initial_state()
    tensor_size = len(initial_state.information_state_tensor(0))
    
    # Benchmark state cloning speed
    print("Benchmarking state clone performance...")
    test_state = game.new_initial_state()
    start = time.time()
    for _ in range(100):
        _ = test_state.clone()
    clone_time = (time.time() - start) / 100 * 1000
    print(f"Average clone time: {clone_time:.2f}ms ({1000/clone_time:.0f} clones/sec)")
    print()

    if test_mode:
        print("🧪 TEST MODE: Running minimal configuration for quick validation\n")
    else:
        print("🚀 PRODUCTION MODE: Full training configuration\n")
    num_iterations = int(training_config["num_iterations"])
    num_traversals = int(training_config["num_traversals"])
    num_val_traversals = int(training_config["num_val_traversals"])
    num_workers = int(training_config["num_workers"])
    if num_workers > 1:
        raise RuntimeError(
            "ESCHER parallel traversal (num_workers > 1) is currently disabled due to stability issues. "
            "Please set num_workers=1 in your configuration."
        )
    batch_size_regret = int(training_config["batch_size_regret"])
    batch_size_value = int(training_config["batch_size_value"])
    batch_size_policy = int(training_config["batch_size_policy"])
    train_steps = int(training_config["train_steps"])
    
    speedup = min(num_workers, 4) if num_workers > 1 else 1
    print(f"Device: {device} | Workers: {num_workers} | Iterations: {num_iterations} | Traversals/iter: {num_traversals}")
    if num_workers > 1:
        print(f"🔥 Parallel mode: {num_workers} CPU workers (expected {speedup}x speedup)")
    print(f"Estimated time per iteration: ~{num_traversals*2//speedup}s (2 players × {num_traversals} trees / {num_workers} workers)")
    print(f"Total estimated time: ~{num_iterations*num_traversals*2/60/speedup:.1f} minutes\n")
    print("Starting training...\n")

    # Setup output directory
    output_dir = PROJECT_ROOT / "results" / "swarm_defense"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {output_dir}\n")
    
    solver = ESCHERSolverTorch(
        game,
        num_iterations=num_iterations,
        num_traversals=num_traversals,
        num_val_fn_traversals=num_val_traversals,
        learning_rate=1e-3,
        batch_size_regret=batch_size_regret,
        batch_size_value=batch_size_value,
        batch_size_average_policy=batch_size_policy,
        policy_network_train_steps=train_steps,
        regret_network_train_steps=train_steps,
        value_network_train_steps=train_steps,
        check_exploitability_every=10,
        compute_exploitability=False,
        save_policy_weights=True,
        infer_device=device,
        train_device=device,
        num_workers=num_workers,  # NEW: parallel traversal
    )
    
    results = solver.solve(save_path_convs=str(output_dir / "swarm"))
    regret_losses, final_policy_loss, convs, nodes, iteration_times = results[:5]
    
    # Save all networks (policy, regret, value)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    checkpoint_dir = output_dir / timestamp
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save policy network
    policy_path = checkpoint_dir / "policy.pt"
    torch.save(solver._policy_network.state_dict(), str(policy_path))
    
    # Save regret networks (one per player)
    regret_paths = []
    for player_id in range(2):  # 2 players in swarm defense
        regret_path = checkpoint_dir / f"regret_player{player_id}.pt"
        torch.save(solver._regret_networks[player_id].state_dict(), str(regret_path))
        regret_paths.append(regret_path)
    
    # Save value network
    value_path = checkpoint_dir / "value.pt"
    torch.save(solver._value_network.state_dict(), str(value_path))
    
    # Capture complete game configuration
    config_obj = getattr(game, "config", None)
    parameter_snapshot: Dict[str, Any]
    if config_obj is not None:
        parameter_snapshot = {
            "grid_rows": config_obj.grid_rows,
            "grid_cols": config_obj.grid_cols,
            "num_targets": config_obj.num_targets,
            "num_ad_units": config_obj.num_ad_units,
            "num_attacking_drones": config_obj.num_attacking_drones,
            "num_interceptors": config_obj.num_interceptors,
            "ad_kill_rate": config_obj.ad_kill_rate,
            "interceptor_reward": config_obj.interceptor_reward,
        }
    else:
        parameter_snapshot = dict(game_params)
    
    game_config = {
        "parameters": parameter_snapshot,
        "requested_parameters": game_params,
        "TOT_CHOICES": list(swarm_defense_game.TOT_CHOICES),
        "TARGET_VALUE_OPTIONS": list(swarm_defense_game.TARGET_VALUE_OPTIONS),
        "AD_COVERAGE_RADIUS": swarm_defense_game.AD_COVERAGE_RADIUS,
        "num_phases": len(swarm_defense_game.Phase),
        "phase_names": [p.name for p in swarm_defense_game.Phase],
        "state_tensor_size": tensor_size,
        "value_network_input_size": tensor_size * 2,
        "num_distinct_actions": game.num_distinct_actions(),
    }
    
    # Save metadata (hyperparameters, training info, and game config)
    metadata = {
        "timestamp": timestamp,
        "num_iterations": num_iterations,
        "num_traversals": num_traversals,
        "num_val_traversals": num_val_traversals,
        "num_workers": num_workers,
        "batch_size_regret": batch_size_regret,
        "batch_size_value": batch_size_value,
        "batch_size_policy": batch_size_policy,
        "train_steps": train_steps,
        "total_nodes": solver.get_num_nodes(),
        "total_time_seconds": sum(iteration_times),
        "final_exploitability": convs[-1] if convs else None,
        "device": device,
        "num_cpu_threads": num_cpu_threads,
        "test_mode": test_mode,
        "config_file": str(args.config) if args.config else None,
        "training_overrides": training_overrides,
        "game_config": game_config,  # Complete game configuration
    }
    metadata_path = checkpoint_dir / "metadata.pt"
    torch.save(metadata, str(metadata_path))
    
    print(f"\nTraining complete!")
    print(f"Total trees traversed: {solver.get_num_nodes():,}")
    print(f"Final exploitability: {convs[-1]:.4f}" if convs else "N/A")
    print(f"Total time: {sum(iteration_times):.1f}s")
    print(f"\n📁 Saved all networks to: {checkpoint_dir}/")
    print(f"   - Policy network: policy.pt")
    print(f"   - Regret networks: regret_player0.pt, regret_player1.pt")
    print(f"   - Value network: value.pt")
    print(f"   - Metadata + Game Config: metadata.pt")
    print(f"       → State size: {tensor_size}, Actions: {game.num_distinct_actions()}, Phases: {len(swarm_defense_game.Phase)}")
    if convs:
        print(f"   - Exploitability curve: {output_dir}/swarm_convs.npy")


if __name__ == "__main__":
    main()
