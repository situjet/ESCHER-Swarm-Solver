import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import pyspiel

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

from ESCHER_Torch import ESCHERSolverTorch


def select_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    output_dir = PROJECT_ROOT / "results" / "leduc"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_torch_device()
    game = pyspiel.load_game("leduc_poker")

    solver = ESCHERSolverTorch(
        game,
        num_iterations=50,
        num_traversals=130000,
        num_val_fn_traversals=50,
        learning_rate=1e-3,
        batch_size_regret=128,
        batch_size_value=128,
        batch_size_average_policy=128,
        policy_network_train_steps=200,
        regret_network_train_steps=200,
        value_network_train_steps=200,
        check_exploitability_every=1,
        save_regret_networks=None,
        save_policy_weights=False,
        infer_device=device,
        train_device=device,
        clear_value_buffer=True,
        importance_sampling=True,
    )

    print(f"Using Torch device: {device}")
    print(f"Saving outputs to: {output_dir}")

    (
        regret_losses,
        final_policy_loss,
        convs,
        nodes,
        iteration_times,
        _policy_loss_history,
        _value_loss_history,
    ) = solver.solve(
        save_path_convs=str(output_dir / "eschersolver")
    )

    print("Run complete.")
    print(f"Final policy loss: {final_policy_loss}")
    print(f"Stored exploitabilities: {convs}")
    print(f"Node counts per checkpoint: {nodes}")

    iterations = np.arange(1, len(convs) + 1)

    convs_path = output_dir / "nash_conv.png"
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, convs, marker="o", linewidth=1.5)
    plt.xlabel("Iteration")
    plt.ylabel("NashConv")
    plt.title("ESCHER-Torch Exploitability on Leduc Poker")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(convs_path)
    plt.close()

    times_path = output_dir / "iteration_times.png"
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, iteration_times, marker="o", linewidth=1.5, color="tab:orange")
    plt.xlabel("Iteration")
    plt.ylabel("Iteration Duration (s)")
    plt.title("ESCHER-Torch Iteration Runtime on Leduc Poker")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(times_path)
    plt.close()

    np.save(output_dir / "nash_conv.npy", np.array(convs))
    np.save(output_dir / "iteration_times.npy", np.array(iteration_times))

    print(f"Saved NashConv curve to: {convs_path}")
    print(f"Saved iteration timing curve to: {times_path}")


if __name__ == "__main__":
    main()
