"""ESCHER-Torch entry point for Swarm Defense Large v2.

This runner wires the PyTorch ESCHER solver to the Swarm-AD-Large-OpenSpiel-2
environment. The large-scale game does not expose an information-state tensor
by default, so this script injects a deterministic encoder based on the
`SwarmDefenseLargeState.snapshot()` structure before launching training.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pyspiel
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
SWARM_REPO_PATH = PROJECT_ROOT.parent / "Swarm-AD-Large-OpenSpiel-2"
if str(SWARM_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(SWARM_REPO_PATH))

import swarm_defense_large_game as large_swarm  # noqa: E402

from ESCHER_Torch import ESCHERSolverTorch  # noqa: E402

GAME_NAME = "swarm_defense_large_v2"
RESULT_DIR_NAME = "swarm_defense_large_v2"
ENCODER_VERSION = "swarm_large_snapshot_v1"
NETWORK_LAYERS = (512, 256, 128)
VALUE_NETWORK_LAYERS = (512, 256, 128)

NUM_TARGETS = large_swarm.NUM_TARGETS
NUM_AD_UNITS = large_swarm.NUM_AD_UNITS
NUM_DRONES = large_swarm.NUM_ATTACKING_DRONES
NUM_INTERCEPTORS = large_swarm.NUM_INTERCEPTORS
NUM_WAVES = large_swarm.NUM_WAVES
ARENA_HEIGHT = large_swarm.ARENA_HEIGHT
ARENA_WIDTH = large_swarm.ARENA_WIDTH
MAX_TARGET_VALUE = max(large_swarm.TARGET_VALUE_OPTIONS)
MAX_TOT_VALUE = max(large_swarm.LARGE_TOT_CHOICES)
TOT_INDEX_DENOM = max(1, len(large_swarm.LARGE_TOT_CHOICES) - 1)
BLUEPRINT_INDEX_DENOM = max(1, len(large_swarm.MIDPOINT_STRATEGIES) - 1)
TOTAL_TARGET_SLOTS = NUM_TARGETS + NUM_AD_UNITS
MAX_RETURNS = NUM_DRONES * MAX_TARGET_VALUE + NUM_INTERCEPTORS * large_swarm.INTERCEPTOR_LEFTOVER_VALUE
MAX_DISTANCE = math.hypot(ARENA_HEIGHT, ARENA_WIDTH) + large_swarm.WAVE_TRANSITION_BUFFER
MAX_TIME = MAX_DISTANCE / max(large_swarm.DRONE_SPEED, 1e-5) + large_swarm.WAVE_TRANSITION_BUFFER
PHASE_TO_INDEX = {phase.name: idx for idx, phase in enumerate(large_swarm.Phase)}
DESTROYED_TO_INDEX = {
    "none": 0,
    "ad": 1,
    "ad_target": 2,
    "interceptor": 3,
    "drone": 4,
    "target": 5,
    "other": 6,
}

DEFAULT_CONFIG = {
    "num_iterations": 10,
    "num_traversals": 100,
    "num_val_traversals": 10,
    "num_workers": 1,
    "batch_size_regret": 512,
    "batch_size_value": 128,
    "batch_size_policy": 512,
    "train_steps": 200,
    "learning_rate": 1e-3,
}

FAST_CONFIG = {
    "num_iterations": 2,
    "num_traversals": 8,
    "num_val_traversals": 2,
    "num_workers": 1,
    "batch_size_regret": 32,
    "batch_size_value": 32,
    "batch_size_policy": 32,
    "train_steps": 40,
    "learning_rate": 3e-4,
}


def _norm(value: Optional[float], denom: float) -> float:
    if value is None:
        return 0.0
    if denom == 0:
        return float(value)
    return float(value) / float(denom)


def _encode_phase(phase_name: str) -> List[float]:
    vec = [0.0] * len(PHASE_TO_INDEX)
    idx = PHASE_TO_INDEX.get(phase_name, 0)
    vec[idx] = 1.0
    return vec


def _encode_discovered_ads(discovered: Sequence[int]) -> List[float]:
    vec = [0.0] * NUM_AD_UNITS
    for idx in discovered:
        if 0 <= idx < NUM_AD_UNITS:
            vec[idx] = 1.0
    return vec


def _encode_targets(snapshot: Dict[str, object]) -> List[float]:
    targets = list(snapshot.get("targets", ()))
    destroyed_flags = list(snapshot.get("target_destroyed", ()))
    features: List[float] = []
    for idx in range(NUM_TARGETS):
        if idx < len(targets):
            target = targets[idx]
            row = getattr(target, "row", 0.0)
            col = getattr(target, "col", 0.0)
            value = getattr(target, "value", 0.0)
            present = 1.0
            destroyed = 1.0 if idx < len(destroyed_flags) and destroyed_flags[idx] else 0.0
        else:
            row = col = value = 0.0
            present = destroyed = 0.0
        features.extend(
            [
                present,
                _norm(row, ARENA_HEIGHT),
                _norm(col, ARENA_WIDTH),
                _norm(value, MAX_TARGET_VALUE),
                destroyed,
            ]
        )
    return features


def _encode_ad_units(snapshot: Dict[str, object], discovered: Sequence[int]) -> List[float]:
    ad_units = list(snapshot.get("ad_units", ()))
    discovered_set = set(discovered)
    features: List[float] = []
    for idx in range(NUM_AD_UNITS):
        if idx < len(ad_units):
            unit = ad_units[idx]
            row, col = unit.get("position", (0.0, 0.0))
            alive = 1.0 if unit.get("alive", False) else 0.0
            orientation = float(unit.get("orientation", 0.0))
            present = 1.0
        else:
            row = col = orientation = 0.0
            alive = present = 0.0
        features.extend(
            [
                present,
                _norm(row, ARENA_HEIGHT),
                _norm(col, ARENA_WIDTH),
                alive,
                math.cos(orientation),
                math.sin(orientation),
                1.0 if idx in discovered_set else 0.0,
            ]
        )
    return features


def _encode_destroyed_type(label: Optional[str]) -> List[float]:
    vec = [0.0] * len(DESTROYED_TO_INDEX)
    if label:
        prefix = str(label).split(":", 1)[0]
        idx = DESTROYED_TO_INDEX.get(prefix, DESTROYED_TO_INDEX["other"])
    else:
        idx = DESTROYED_TO_INDEX["none"]
    vec[idx] = 1.0
    return vec


def _encode_drone_plan(plan: Optional[Dict[str, object]]) -> List[float]:
    if plan is None:
        entry = (0.0, 0.0)
        destination = (0.0, 0.0)
        tot_idx = 0
        tot_value = 0.0
        blueprint_idx = 0
        hold_time = arrival_time = total_distance = 0.0
        wave = 1
        destroyed_by = None
        strike_success = None
        damage = 0.0
        target_idx = -1
        target_value = 0.0
        interceptor_hit = None
        interceptor_time = None
        intercepts = ()
    else:
        entry = plan.get("entry") or (0.0, 0.0)
        destination = plan.get("destination") or (0.0, 0.0)
        tot_idx = int(plan.get("tot_idx", 0))
        tot_value = float(plan.get("tot", 0.0))
        blueprint_idx = int(plan.get("blueprint_idx", 0))
        hold_time = float(plan.get("hold_time", 0.0))
        arrival_time = float(plan.get("arrival_time", 0.0))
        total_distance = float(plan.get("total_distance", 0.0))
        wave = int(plan.get("wave", 1))
        destroyed_by = plan.get("destroyed_by")
        strike_success = plan.get("strike_success")
        damage = float(plan.get("damage_inflicted", 0.0))
        target_idx = int(plan.get("target_idx", -1))
        target_value = plan.get("target_value")
        interceptor_hit = plan.get("interceptor_hit")
        interceptor_time = plan.get("interceptor_time")
        intercepts = plan.get("intercepts", ())
    assigned = 0.0 if plan is None else 1.0
    entry_row, entry_col = entry
    dest_row, dest_col = destination
    target_slot = max(target_idx, 0)
    target_is_ad = 1.0 if target_idx >= NUM_TARGETS and target_idx >= 0 else 0.0
    strike_flag = 0.0 if strike_success is None else 1.0
    strike_value = 1.0 if strike_success else 0.0
    destroy_one_hot = _encode_destroyed_type(destroyed_by)
    interceptor_flag = 1.0 if interceptor_hit else 0.0
    if interceptor_hit:
        interceptor_row, interceptor_col = interceptor_hit
    else:
        interceptor_row = interceptor_col = 0.0
    intercept_count = len(intercepts) if intercepts else 0
    features = [
        assigned,
        _norm(entry_row, ARENA_HEIGHT),
        _norm(entry_col, ARENA_WIDTH),
        _norm(dest_row, ARENA_HEIGHT),
        _norm(dest_col, ARENA_WIDTH),
        _norm(target_slot, max(1, TOTAL_TARGET_SLOTS)),
        target_is_ad,
        _norm(tot_idx, TOT_INDEX_DENOM),
        _norm(tot_value, MAX_TOT_VALUE),
        _norm(blueprint_idx, BLUEPRINT_INDEX_DENOM),
        _norm(wave, max(1, NUM_WAVES)),
        _norm(hold_time, MAX_TIME),
        _norm(arrival_time, MAX_TIME),
        _norm(total_distance, MAX_DISTANCE),
        1.0 if destroyed_by else 0.0,
    ]
    features.extend(destroy_one_hot)
    features.extend(
        [
            strike_flag,
            strike_value,
            _norm(damage, MAX_TARGET_VALUE),
            _norm(intercept_count, max(1, NUM_AD_UNITS)),
            interceptor_flag,
            _norm(interceptor_row, ARENA_HEIGHT),
            _norm(interceptor_col, ARENA_WIDTH),
            _norm(interceptor_time, MAX_TIME),
            _norm(target_value, MAX_TARGET_VALUE),
        ]
    )
    return features


def _encode_drones(snapshot: Dict[str, object]) -> List[float]:
    drones = list(snapshot.get("drones", ()))
    features: List[float] = []
    for idx in range(NUM_DRONES):
        plan = drones[idx] if idx < len(drones) else None
        features.extend(_encode_drone_plan(plan))
    return features


def _encode_returns(snapshot: Dict[str, object], player: int) -> List[float]:
    returns = snapshot.get("returns", (0.0, 0.0))
    attacker = float(returns[0]) if len(returns) > 0 else 0.0
    defender = float(returns[1]) if len(returns) > 1 else 0.0
    player_return = float(returns[player]) if player < len(returns) else attacker
    return [
        _norm(attacker, MAX_RETURNS),
        _norm(defender, MAX_RETURNS),
        _norm(player_return, MAX_RETURNS),
    ]


def _snapshot_to_tensor(snapshot: Dict[str, object], player: int) -> List[float]:
    features: List[float] = []
    features.extend(_encode_phase(snapshot.get("phase", "")))
    features.append(_norm(snapshot.get("current_wave", 1), max(1, NUM_WAVES)))
    features.append(_norm(snapshot.get("remaining_drones", NUM_DRONES), max(1, NUM_DRONES)))
    features.append(
        _norm(snapshot.get("remaining_interceptors", NUM_INTERCEPTORS), max(1, NUM_INTERCEPTORS))
    )
    discovered = snapshot.get("discovered_ads", ())
    features.extend(_encode_discovered_ads(discovered))
    features.extend(_encode_targets(snapshot))
    features.extend(_encode_ad_units(snapshot, discovered))
    features.extend(_encode_drones(snapshot))
    features.extend(_encode_returns(snapshot, player))
    return features


def ensure_information_state_tensor() -> None:
    state_cls = large_swarm.SwarmDefenseLargeState
    if "_apply_action" not in state_cls.__dict__:
        def _apply_action(self, action: int) -> None:  # type: ignore[override]
            self.apply_action(action)

        setattr(state_cls, "_apply_action", _apply_action)

    if "_legal_actions" not in state_cls.__dict__:
        def _legal_actions(self, player: Optional[int] = None) -> List[int]:  # type: ignore[override]
            return self.legal_actions(player)

        setattr(state_cls, "_legal_actions", _legal_actions)

    if "information_state_tensor" in state_cls.__dict__:
        return

    def _information_state_tensor(self, player: int) -> List[float]:  # type: ignore[override]
        return _snapshot_to_tensor(self.snapshot(), player)

    setattr(state_cls, "information_state_tensor", _information_state_tensor)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ESCHER-Torch on Swarm Defense Large v2")
    parser.add_argument("--fast-test", action="store_true", help="Use the lightweight configuration")
    parser.add_argument("--iterations", type=int, default=None, help="Override number of iterations")
    parser.add_argument("--traversals", type=int, default=None, help="Override traversals per player")
    parser.add_argument("--val-traversals", type=int, default=None, help="Override value traversals")
    parser.add_argument("--workers", type=int, default=None, help="Number of tree traversal workers")
    parser.add_argument("--batch-regret", type=int, default=None, help="Regret batch size override")
    parser.add_argument("--batch-value", type=int, default=None, help="Value batch size override")
    parser.add_argument("--batch-policy", type=int, default=None, help="Policy batch size override")
    parser.add_argument("--train-steps", type=int, default=None, help="Training steps per network")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional custom results directory")
    parser.add_argument("--cpu-threads", type=int, default=1, help="Torch intra/inter-op thread count")
    parser.add_argument("--force-cpu", action="store_true", help="Disable CUDA even if available")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Dict[str, float]:
    base = FAST_CONFIG.copy() if args.fast_test else DEFAULT_CONFIG.copy()
    overrides = {
        "num_iterations": args.iterations,
        "num_traversals": args.traversals,
        "num_val_traversals": args.val_traversals,
        "num_workers": args.workers,
        "batch_size_regret": args.batch_regret,
        "batch_size_value": args.batch_value,
        "batch_size_policy": args.batch_policy,
        "train_steps": args.train_steps,
        "learning_rate": args.lr,
    }
    for key, value in overrides.items():
        if value is not None:
            base[key] = value
    return base


def main() -> None:
    args = parse_args()
    config = build_config(args)

    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    cpu_threads = max(1, args.cpu_threads)
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(cpu_threads)

    ensure_information_state_tensor()

    game = pyspiel.load_game(GAME_NAME)
    initial_state = game.new_initial_state()
    tensor_size = len(initial_state.information_state_tensor(0))

    num_workers = max(1, int(config["num_workers"]))
    est_time = max(1, int(config["num_traversals"])) * 2
    if num_workers > 1:
        est_time = est_time // min(num_workers, 4)

    print("Swarm Defense Large v2 | ESCHER-Torch")
    print(f"Device: {device} | CPU threads: {cpu_threads} | State dims: {tensor_size}")
    print(
        "Iterations: {iters} | Traversals/player: {trav} | Value traversals: {val}".format(
            iters=config["num_iterations"], trav=config["num_traversals"], val=config["num_val_traversals"]
        )
    )
    print(
        "Batches (regret/value/policy): {r}/{v}/{p} | Train steps: {steps}".format(
            r=config["batch_size_regret"],
            v=config["batch_size_value"],
            p=config["batch_size_policy"],
            steps=config["train_steps"],
        )
    )
    print(
        f"Workers: {num_workers} | Approx traverse time hint: ~{est_time}s per iteration (heuristic)"
    )

    output_dir = args.output_dir or (PROJECT_ROOT / "results" / RESULT_DIR_NAME)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {output_dir}")

    solver = ESCHERSolverTorch(
        game,
        policy_network_layers=NETWORK_LAYERS,
        regret_network_layers=NETWORK_LAYERS,
        value_network_layers=VALUE_NETWORK_LAYERS,
        num_iterations=int(config["num_iterations"]),
        num_traversals=int(config["num_traversals"]),
        num_val_fn_traversals=int(config["num_val_traversals"]),
        learning_rate=float(config["learning_rate"]),
        batch_size_regret=int(config["batch_size_regret"]),
        batch_size_value=int(config["batch_size_value"]),
        batch_size_average_policy=int(config["batch_size_policy"]),
        policy_network_train_steps=int(config["train_steps"]),
        regret_network_train_steps=int(config["train_steps"]),
        value_network_train_steps=int(config["train_steps"]),
        check_exploitability_every=max(1, int(config["num_iterations"]) // 2),
        compute_exploitability=False,
        infer_device=device,
        train_device=device,
        save_policy_weights=True,
        memory_capacity=int(2e5),
        num_workers=num_workers,
        all_actions=False,  # Sampling-only evaluation; enumerating all actions is intractable here
    )

    start = time.time()
    results = solver.solve(save_path_convs=str(output_dir / "swarm_large"))
    elapsed = time.time() - start
    regret_losses, final_policy_loss, convs, nodes, iteration_times = results[:5]

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    checkpoint_dir = output_dir / timestamp
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.save(solver._policy_network.state_dict(), str(checkpoint_dir / "policy.pt"))
    for player_id in range(2):
        torch.save(
            solver._regret_networks[player_id].state_dict(),
            str(checkpoint_dir / f"regret_player{player_id}.pt"),
        )
    torch.save(solver._value_network.state_dict(), str(checkpoint_dir / "value.pt"))

    metadata = {
        "timestamp": timestamp,
        "encoder_version": ENCODER_VERSION,
        "tensor_size": tensor_size,
        "config": config,
        "total_time_seconds": elapsed,
        "iteration_times": iteration_times,
        "regret_losses": {k: v for k, v in regret_losses.items()},
        "final_policy_loss": final_policy_loss,
        "convs": convs,
        "nodes": nodes,
        "device": device,
        "cpu_threads": cpu_threads,
    }
    torch.save(metadata, str(checkpoint_dir / "metadata.pt"))

    print("Run complete.")
    print(f"Checkpoints written to: {checkpoint_dir}")
    print(f"Elapsed time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
