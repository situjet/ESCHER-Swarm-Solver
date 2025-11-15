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
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import pyspiel

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT.parent / "Swarm-AD-OpenSpiel"))

import swarm_defense_game  # noqa: F401
from ESCHER_Torch import ESCHERSolverTorch


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set PyTorch CPU threads early (before any torch operations)
    # This allows PyTorch operations to use multiple CPU cores
    num_cpu_threads = 1  # Adjust based on your CPU cores
    torch.set_num_threads(num_cpu_threads)
    torch.set_num_interop_threads(num_cpu_threads)
    print(f"🔧 PyTorch CPU threads: {num_cpu_threads}")
    
    game = pyspiel.load_game("swarm_defense")
        
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

    # Configuration
    TEST_MODE = False  # Set True for quick testing (2 min), False for production (30-60 min)
    
    if TEST_MODE:
        print("🧪 TEST MODE: Running minimal configuration for quick validation\n")
        num_iterations = 3
        num_traversals = 10
        num_val_traversals = 5
        num_workers = 1
        batch_size_regret = 60
        batch_size_value = 60
        batch_size_policy = 60
        train_steps = 50
    else:
        print("🚀 PRODUCTION MODE: Full training configuration\n")
        num_iterations = 8
        num_traversals = 100
        num_val_traversals = 10
        num_workers = 1  # Parallel tree traversal workers (CPU cores)
        batch_size_regret = 50
        batch_size_value = 40  # Reduced: 20 traversals → ~40-80 samples (was 128)
        batch_size_policy = 64  # Reduced to ensure enough samples (was 128)
        train_steps = 200
    
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
    
    # Save metadata (hyperparameters and training info)
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
    print(f"   - Metadata: metadata.pt")
    if convs:
        print(f"   - Exploitability curve: {output_dir}/swarm_convs.npy")


if __name__ == "__main__":
    main()

