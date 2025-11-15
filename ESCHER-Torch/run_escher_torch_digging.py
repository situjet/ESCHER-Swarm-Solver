import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import pyspiel

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent.parent
DIGGING_SRC = REPO_ROOT / "digging_game" / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))
if str(DIGGING_SRC) not in sys.path:
    sys.path.insert(0, str(DIGGING_SRC))

from ESCHER_Torch import ESCHERSolverTorch  # noqa: E402
import hide_seek_game  # type: ignore  # noqa: F401,E402  # Registers the custom game with OpenSpiel


def select_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_solver(device: str) -> ESCHERSolverTorch:
    game = pyspiel.load_game("py_hide_seek")
    return ESCHERSolverTorch(
        game,
        num_iterations=20,
        num_traversals=130,
        num_val_fn_traversals=10,
        learning_rate=1e-3,
        batch_size_regret=1000,
        batch_size_value=200,
        batch_size_average_policy=1000,
        policy_network_train_steps=1500,
        regret_network_train_steps=500,
        value_network_train_steps=400,
        check_exploitability_every=2,
        save_regret_networks=None,
        save_policy_weights=False,
        infer_device=device,
        train_device=device,
        clear_value_buffer=True,
        importance_sampling=True,
        debug_logging=True,
        compute_exploitability=True,
    )


def main() -> None:
    output_dir = PROJECT_ROOT / "results" / "digging"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_torch_device()
    print(f"Using Torch device: {device}")
    print(f"Saving outputs to: {output_dir}")

    attempt = 1
    results = None
    max_attempts = 5
    while results is None and attempt <= max_attempts:
        print(f"=== Attempt {attempt} ===")
        solver = build_solver(device)
        try:
            results = solver.solve(
                save_path_convs=str(output_dir / f"eschersolver_attempt_{attempt}")
            )
        except Exception as exc:  # pragma: no cover - defensive retry loop
            print(f"Attempt {attempt} failed: {exc}")
            attempt += 1
    if results is None:
        raise RuntimeError(f"Failed to complete after {max_attempts} attempts")

    (
        regret_losses,
        final_policy_loss,
        convs,
        nodes,
        iteration_times,
        policy_loss_history,
        value_loss_history,
    ) = results

    iteration_times = np.array(iteration_times, dtype=np.float32)
    policy_loss_array = np.array([
        np.nan if loss is None else float(loss) for loss in policy_loss_history
    ], dtype=np.float32)
    value_loss_array = np.array([
        np.nan if loss is None else float(loss) for loss in value_loss_history
    ], dtype=np.float32)

    if regret_losses:
        max_len = max(len(losses) for losses in regret_losses.values())
        regret_matrix = []
        for losses in regret_losses.values():
            padded = list(losses)
            if len(padded) < max_len:
                padded.extend([np.nan] * (max_len - len(padded)))
            regret_matrix.append(padded)
        regret_mean = np.nanmean(np.array(regret_matrix, dtype=np.float32), axis=0)
    else:
        regret_mean = np.array([], dtype=np.float32)

    np.save(output_dir / "attempt.npy", np.array([attempt], dtype=np.int32))
    np.save(output_dir / "nash_conv.npy", np.array(convs, dtype=np.float32))
    np.save(output_dir / "iteration_times.npy", iteration_times)
    np.save(output_dir / "policy_loss.npy", policy_loss_array)
    np.save(output_dir / "value_loss.npy", value_loss_array)
    if regret_mean.size:
        np.save(output_dir / "regret_loss_mean.npy", regret_mean)

    if convs:
        print(f"Exploitabilities: {convs}")
    print(f"Final policy loss: {final_policy_loss}")
    print(f"Iteration durations: {iteration_times}")
    for player, losses in regret_losses.items():
        print(f"Player {player} regret losses ({len(losses)} samples): {losses}")

    iterations = np.arange(1, len(iteration_times) + 1)

    time_path = output_dir / "iteration_times.png"
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, iteration_times, marker="o", linewidth=1.5, color="tab:orange")
    plt.xlabel("Iteration")
    plt.ylabel("Iteration Duration (s)")
    plt.title("ESCHER-Torch Iteration Runtime on Digging Game")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(time_path)
    plt.close()

    loss_path = output_dir / "value_loss.png"
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, value_loss_array, marker="o", linewidth=1.5, color="tab:blue")
    plt.xlabel("Iteration")
    plt.ylabel("Value Loss")
    plt.title("ESCHER-Torch Value Loss on Digging Game")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(loss_path)
    plt.close()

    policy_path = output_dir / "policy_loss.png"
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, policy_loss_array, marker="o", linewidth=1.5, color="tab:green")
    plt.xlabel("Iteration")
    plt.ylabel("Policy Loss")
    plt.title("ESCHER-Torch Policy Loss on Digging Game")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(policy_path)
    plt.close()

    if regret_mean.size:
        regret_path = output_dir / "regret_loss.png"
        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(1, len(regret_mean) + 1), regret_mean, marker="o", linewidth=1.5, color="tab:red")
        plt.xlabel("Iteration")
        plt.ylabel("Mean Regret Loss")
        plt.title("ESCHER-Torch Regret Loss on Digging Game")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(regret_path)
        plt.close()
        print(f"Saved regret loss curve to: {regret_path}")

    print(f"Saved iteration timing curve to: {time_path}")
    print(f"Saved value loss curve to: {loss_path}")
    print(f"Saved policy loss curve to: {policy_path}")

if __name__ == "__main__":
    main()
