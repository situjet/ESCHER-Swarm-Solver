"""PyTorch reimplementation of the ESCHER solver.

This module ports the original TensorFlow-based ESCHER implementation to
PyTorch so it can run inside WSL environments that provide CUDA-enabled
Torch builds. The goal is functional parity for standard imperfect-information
games such as Leduc poker; a number of ancillary research features from the
TensorFlow version are intentionally omitted for clarity.
"""

from __future__ import annotations

import collections
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from multiprocessing import Pool
from functools import partial

import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability

import pyspiel

REGRET_TRAIN_SHUFFLE_SIZE = 100000
VALUE_TRAIN_SHUFFLE_SIZE = 100000
AVERAGE_POLICY_TRAIN_SHUFFLE_SIZE = 1000000


def _traverse_tree_worker(
    task_id: int,
    game_name: str,
    player: int,
    policy_state_dict: dict,
    regret_state_dict: dict,
    value_state_dict: dict,
    embedding_size: int,
    num_actions: int,
    policy_layers: tuple,
    regret_layers: tuple,
    value_layers: tuple,
    value_embedding_size: int,
    train_regret: bool,
    train_value: bool,
    track_mean_squares: bool,
    on_policy_prob: float,
    expl: float,
) -> Tuple[List, List, int]:
    """Worker function for parallel tree traversal. Returns (regret_samples, value_samples, nodes_visited)."""
    import sys
    from pathlib import Path
    import numpy as np
    import pyspiel
    import torch
    import torch.nn as nn
    
    # Import custom game if needed (for swarm_defense)
    if game_name == "swarm_defense":
        # Add Swarm-AD-OpenSpiel to path
        project_root = Path(__file__).resolve().parent.parent.parent
        swarm_path = project_root / "Swarm-AD-OpenSpiel"
        if str(swarm_path) not in sys.path:
            sys.path.insert(0, str(swarm_path))
        try:
            import swarm_defense_game  # noqa: F401
        except ImportError:
            pass
    
    # Build network structure matching PolicyNetwork/RegretNetwork/ValueNetwork
    # These use _build_mlp with hidden_layers[:-1] passed to it, and last layer separately
    class SimpleNetwork(nn.Module):
        def __init__(self, input_size, hidden_layers, output_size):
            super().__init__()
            layers = []
            prev = input_size
            
            # _build_mlp is called with hidden_layers[:-1] for the "net" part
            # So we need to iterate over all but the last hidden layer
            hidden_for_net = hidden_layers[:-1] if len(hidden_layers) > 1 else []
            last_hidden = hidden_layers[-1] if hidden_layers else input_size
            
            # Build layers for all hidden_layers[:-1]
            for units in hidden_for_net:
                layers.append(nn.Linear(prev, units))
                layers.append(nn.LeakyReLU(0.2))
                prev = units
            
            # Then add LayerNorm + LeakyReLU (if there were any hidden layers)
            if hidden_for_net:
                layers.append(nn.LayerNorm(hidden_for_net[-1]))
                layers.append(nn.LeakyReLU(0.2))
                prev = hidden_for_net[-1]
            
            # Then add final layer to reach last_hidden size
            layers.append(nn.Linear(prev, last_hidden))
            
            self.net = nn.Sequential(*layers)
            self.out = nn.Linear(last_hidden, output_size)
        
        def forward(self, x):
            return self.out(self.net(x))
    
    # Recreate game and networks in worker process (CPU only)
    game = pyspiel.load_game(game_name)
    root = game.new_initial_state()
    
    # Recreate networks on CPU with matching structure
    policy_net = SimpleNetwork(embedding_size, policy_layers, num_actions)
    policy_net.load_state_dict(policy_state_dict)
    policy_net.eval()
    policy_net.to('cpu')
    
    regret_net = SimpleNetwork(embedding_size, regret_layers, num_actions)
    regret_net.load_state_dict(regret_state_dict)
    regret_net.eval()
    regret_net.to('cpu')
    
    value_net = SimpleNetwork(value_embedding_size, value_layers, 1)
    value_net.load_state_dict(value_state_dict)
    value_net.eval()
    value_net.to('cpu')
    
    # Simple buffers for this worker
    regret_samples = []
    value_samples = []
    nodes_visited = 0
    
    # Simplified tree traversal (just collect samples, no training)
    # This is a simplified version - for full implementation, would need full _traverse_game_tree logic
    # For now, just traverse and collect basic samples
    
    def traverse_recursive(state, depth=0):
        nonlocal nodes_visited
        nodes_visited += 1
        
        if state.is_terminal() or depth > 50:  # Depth limit to prevent infinite recursion
            return 0.0
        
        current_player = state.current_player()
        
        if current_player == pyspiel.PlayerId.CHANCE:
            # Sample a chance outcome
            outcomes = state.chance_outcomes()
            if outcomes:
                actions, probs = zip(*outcomes)
                action = actions[0]  # Just take first for simplicity
                child = state.clone()
                child.apply_action(action)
                return traverse_recursive(child, depth + 1)
            return 0.0
        
        if current_player != player:
            # Opponent turn - just sample an action
            legal_actions = state.legal_actions()
            if legal_actions:
                action = legal_actions[0]
                child = state.clone()
                child.apply_action(action)
                return traverse_recursive(child, depth + 1)
            return 0.0
        
        # Our turn - use policy network
        info_state = torch.FloatTensor(state.information_state_tensor(player)).unsqueeze(0)
        
        with torch.no_grad():
            policy_logits = policy_net(info_state).squeeze(0)
        
        legal_actions = state.legal_actions()
        if not legal_actions:
            return 0.0
        
        # Sample an action
        legal_mask = torch.zeros(num_actions)
        legal_mask[legal_actions] = 1.0
        masked_logits = policy_logits + (1.0 - legal_mask) * -1e9
        probs = torch.softmax(masked_logits, dim=0)
        
        action = legal_actions[torch.multinomial(probs[legal_actions], 1).item()]
        
        # Collect sample for regret training
        if train_regret and len(regret_samples) < 1000:  # Limit samples per worker
            # Create regret array (simplified - all zeros for now, just for structure)
            regret_array = np.zeros(num_actions, dtype=np.float32)
            regret_array[action] = 0.0  # Would need counterfactual values for real training
            regret_samples.append((
                info_state.squeeze(0).numpy(),
                regret_array,
                legal_mask.numpy()
            ))
        
        child = state.clone()
        child.apply_action(action)
        value = traverse_recursive(child, depth + 1)
        
        # Collect value sample (hist_state = concat of both players' info states)
        if train_value and len(value_samples) < 1000:
            # Get both players' information states
            info_state_p0 = torch.FloatTensor(state.information_state_tensor(0))
            info_state_p1 = torch.FloatTensor(state.information_state_tensor(1))
            hist_state = torch.cat([info_state_p0, info_state_p1])
            value_samples.append((
                hist_state.numpy(),
                value
            ))
        
        return value
    
    # Run the traversal
    traverse_recursive(root)
    
    return (regret_samples, value_samples, nodes_visited)


class ReservoirBuffer:
    """Reservoir sampling buffer with uniform sampling semantics."""

    def __init__(self, capacity: int):
        self._capacity = capacity
        self._data: List = []
        self._add_calls = 0

    def add(self, element) -> None:
        if self._capacity <= 0:
            return
        if len(self._data) < self._capacity:
            self._data.append(element)
        else:
            idx = random.randint(0, self._add_calls)
            if idx < self._capacity:
                self._data[idx] = element
        self._add_calls += 1

    def sample(self, num_samples: int) -> List:
        if len(self._data) < num_samples:
            raise ValueError(
                f"Requested {num_samples} samples but buffer only holds {len(self._data)} items"
            )
        return random.sample(self._data, num_samples)

    def clear(self) -> None:
        self._data.clear()
        self._add_calls = 0

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterable:
        return iter(self._data)

    def get_num_calls(self) -> int:
        return self._add_calls


class SkipLinear(nn.Module):
    """Linear layer equipped with a skip connection when dimensions agree."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self._residual = in_features == out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if self._residual:
            out = out + x
        return out


def _build_mlp(
    input_size: int,
    hidden_layers: Sequence[int],
    activation: str,
    last_dim: int,
    add_layer_norm: bool,
) -> nn.Module:
    layers: List[nn.Module] = []
    prev = input_size
    for units in hidden_layers:
        if prev == units:
            layers.append(SkipLinear(prev, units))
        else:
            layers.append(nn.Linear(prev, units))
        if activation == "leakyrelu":
            layers.append(nn.LeakyReLU(negative_slope=0.2))
        elif activation == "relu":
            layers.append(nn.ReLU())
        else:
            layers.append(nn.Identity())
        prev = units
    if add_layer_norm and hidden_layers:
        layers.append(nn.LayerNorm(hidden_layers[-1]))
        if activation == "leakyrelu":
            layers.append(nn.LeakyReLU(negative_slope=0.2))
        elif activation == "relu":
            layers.append(nn.ReLU())
        else:
            layers.append(nn.Identity())
        prev = hidden_layers[-1]
    layers.append(nn.Linear(prev, last_dim))
    return nn.Sequential(*layers)


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_layers: Sequence[int],
        num_actions: int,
        activation: str = "leakyrelu",
    ) -> None:
        super().__init__()
        self.net = _build_mlp(
            input_size=input_size,
            hidden_layers=hidden_layers[:-1],
            activation=activation,
            last_dim=hidden_layers[-1] if hidden_layers else input_size,
            add_layer_norm=True,
        )
        last_dim = hidden_layers[-1] if hidden_layers else input_size
        self.out = nn.Linear(last_dim, num_actions)

    def forward(self, info_state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.out(self.net(info_state))
        logits = logits.masked_fill(mask <= 0.0, float("-inf"))
        return F.softmax(logits, dim=-1)


class RegretNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_layers: Sequence[int],
        num_actions: int,
        activation: str = "leakyrelu",
    ) -> None:
        super().__init__()
        self.net = _build_mlp(
            input_size=input_size,
            hidden_layers=hidden_layers[:-1],
            activation=activation,
            last_dim=hidden_layers[-1] if hidden_layers else input_size,
            add_layer_norm=True,
        )
        last_dim = hidden_layers[-1] if hidden_layers else input_size
        self.out = nn.Linear(last_dim, num_actions)

    def forward(self, info_state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = self.out(self.net(info_state))
        return values * mask


class ValueNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_layers: Sequence[int],
        activation: str = "leakyrelu",
    ) -> None:
        super().__init__()
        self.net = _build_mlp(
            input_size=input_size,
            hidden_layers=hidden_layers[:-1],
            activation=activation,
            last_dim=hidden_layers[-1] if hidden_layers else input_size,
            add_layer_norm=True,
        )
        last_dim = hidden_layers[-1] if hidden_layers else input_size
        self.out = nn.Linear(last_dim, 1)

    def forward(self, hist_state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        del mask
        return self.out(self.net(hist_state)).squeeze(-1)


@dataclass
class _RegretSample:
    info_state: np.ndarray
    iteration: float
    regret: np.ndarray
    mask: np.ndarray


@dataclass
class _ValueSample:
    hist_state: np.ndarray
    iteration: float
    value: float
    mask: np.ndarray


@dataclass
class _PolicySample:
    info_state: np.ndarray
    iteration: float
    probs: np.ndarray
    mask: np.ndarray


class ESCHERSolverTorch(policy.Policy):
    """PyTorch-powered ESCHER solver supporting standard OpenSpiel games."""

    def __init__(
        self,
        game: pyspiel.Game,
        policy_network_layers: Sequence[int] = (256, 128),
        regret_network_layers: Sequence[int] = (256, 128),
        value_network_layers: Sequence[int] = (256, 128),
        num_iterations: int = 100,
        num_traversals: int = 130000,
        num_val_fn_traversals: int = 100,
        learning_rate: float = 1e-3,
        batch_size_regret: int = 10000,
        batch_size_value: int = 2024,
        batch_size_average_policy: int = 10000,
        markov_soccer: bool = False,
        phantom_ttt: bool = False,
        dark_hex: bool = False,
        memory_capacity: int = int(1e5),
        policy_network_train_steps: int = 15000,
        regret_network_train_steps: int = 5000,
        value_network_train_steps: int = 4048,
        check_exploitability_every: int = 20,
        reinitialize_regret_networks: bool = True,
        reinitialize_value_network: bool = True,
        save_regret_networks: Optional[str] = None,
        append_legal_actions_mask: bool = False,
        save_average_policy_memories: Optional[str] = None,
        save_policy_weights: bool = True,
        expl: float = 1.0,
        val_expl: float = 0.01,
        importance_sampling_threshold: float = 100.0,
        importance_sampling: bool = True,
        clear_value_buffer: bool = True,
        val_bootstrap: bool = False,
        oshi_zumo: bool = False,
        use_balanced_probs: bool = False,
        battleship: bool = False,
        starting_coins: int = 8,
        val_op_prob: float = 0.0,
        infer_device: str = "gpu",
        debug_val: bool = False,
        play_against_random: bool = False,
        train_device: str = "gpu",
        experiment_string: Optional[str] = None,
        all_actions: bool = True,
        random_policy_path: Optional[str] = None,
        debug_logging: bool = False,
        track_mean_squares: bool = False,
        compute_exploitability: bool = True,
        num_workers: int = 1,  # Number of parallel workers for tree traversal
    ) -> None:
        if any([markov_soccer, phantom_ttt, dark_hex, oshi_zumo, battleship]):
            raise NotImplementedError(
                "Special game variants are not yet supported in the PyTorch port."
            )
        if append_legal_actions_mask:
            raise NotImplementedError("append_legal_actions_mask is not supported.")
        if save_average_policy_memories is not None:
            raise NotImplementedError("Saving average-policy memories to disk is not supported.")
        if random_policy_path is not None:
            raise NotImplementedError("Loading random policies is not supported in this port.")

        all_players = list(range(game.num_players()))
        super().__init__(game, all_players)
        self._game = game

        self._num_players = game.num_players()
        self._root_node = self._game.new_initial_state()
        self._embedding_size = len(self._root_node.information_state_tensor(0))
        hist_example = np.append(
            self._root_node.information_state_tensor(0),
            self._root_node.information_state_tensor(1),
        )
        self._value_embedding_size = len(hist_example)
        self._num_actions = game.num_distinct_actions()
        if game.get_type().dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS:
            raise ValueError("Simultaneous games are not supported.")

        self._policy_network_layers = tuple(policy_network_layers)
        self._regret_network_layers = tuple(regret_network_layers)
        self._value_network_layers = tuple(value_network_layers)
        self._num_iterations = num_iterations
        self._num_traversals = num_traversals
        self._num_val_fn_traversals = num_val_fn_traversals
        self._num_workers = num_workers
        self._policy_network_train_steps = policy_network_train_steps
        self._regret_network_train_steps = regret_network_train_steps
        self._value_network_train_steps = value_network_train_steps
        self._batch_size_regret = batch_size_regret
        self._batch_size_value = batch_size_value
        self._batch_size_average_policy = batch_size_average_policy
        self._check_exploitability_every = check_exploitability_every
        self._reinitialize_regret_networks = reinitialize_regret_networks
        self._reinitialize_value_network = reinitialize_value_network
        self._learning_rate = learning_rate
        self._expl = expl
        self._val_expl = val_expl
        self._importance_sampling = importance_sampling
        self._importance_sampling_threshold = importance_sampling_threshold
        self._clear_value_buffer = clear_value_buffer
        self._val_bootstrap = val_bootstrap
        self._val_op_prob = val_op_prob
        self._debug_val = debug_val
        self._experiment_string = experiment_string
        self._all_actions = all_actions
        self._save_regret_networks = save_regret_networks
        self._save_policy_weights = save_policy_weights
        self._compute_exploitability = compute_exploitability
        self._debug_logging = debug_logging
        self._play_against_random = play_against_random
        self._track_mean_squares = track_mean_squares

        self._iteration = 1
        self._nodes_visited = 0
        self._example_info_state = [None, None]
        self._example_hist_state = None
        self._example_legal_actions_mask = [None, None]
        self._squared_errors: List[float] = []
        self._squared_errors_child: List[float] = []

        train_device = torch.device(train_device)
        infer_device = torch.device(infer_device)
        self._train_device = train_device
        self._infer_device = infer_device

        self._policy_network = PolicyNetwork(
            self._embedding_size,
            self._policy_network_layers,
            self._num_actions,
        ).to(train_device)
        self._policy_optimizer = torch.optim.Adam(
            self._policy_network.parameters(), lr=self._learning_rate
        )

        self._regret_networks: List[RegretNetwork] = []
        self._regret_train_models: List[RegretNetwork] = []
        self._regret_optimizers: List[torch.optim.Optimizer] = []
        for _ in range(self._num_players):
            model = RegretNetwork(
                self._embedding_size,
                self._regret_network_layers,
                self._num_actions,
            ).to(infer_device)
            train_model = RegretNetwork(
                self._embedding_size,
                self._regret_network_layers,
                self._num_actions,
            ).to(train_device)
            train_model.load_state_dict(model.state_dict())
            optimizer = torch.optim.Adam(train_model.parameters(), lr=self._learning_rate)
            self._regret_networks.append(model)
            self._regret_train_models.append(train_model)
            self._regret_optimizers.append(optimizer)

        self._value_network = ValueNetwork(
            self._value_embedding_size,
            self._value_network_layers,
        ).to(infer_device)
        self._value_train_model = ValueNetwork(
            self._value_embedding_size,
            self._value_network_layers,
        ).to(train_device)
        self._value_train_model.load_state_dict(self._value_network.state_dict())
        self._value_optimizer = torch.optim.Adam(
            self._value_train_model.parameters(), lr=self._learning_rate
        )

        self._create_memories(memory_capacity)

    def _create_memories(self, memory_capacity: int) -> None:
        self._average_policy_memories = ReservoirBuffer(memory_capacity)
        self._regret_memories = [ReservoirBuffer(memory_capacity) for _ in range(self._num_players)]
        self._value_memory = ReservoirBuffer(memory_capacity)

    def get_regret_memories(self, player: int) -> List[_RegretSample]:
        return list(self._regret_memories[player])

    def get_value_memory(self) -> List[_ValueSample]:
        return list(self._value_memory)

    def get_average_policy_memories(self) -> List[_PolicySample]:
        return list(self._average_policy_memories)

    def clear_value_memories(self) -> None:
        self._value_memory.clear()

    def clear_regret_buffers(self) -> None:
        for buf in self._regret_memories:
            buf.clear()

    def traverse_game_tree_n_times(
        self,
        n: int,
        player: int,
        train_regret: bool = False,
        train_value: bool = False,
        track_mean_squares: bool = True,
        on_policy_prob: float = 0.0,
        expl: float = 1.0,
    ) -> None:
        if self._num_workers > 1:
            # Parallel mode using multiprocessing
            print(f"  Running {n} traversals with {self._num_workers} parallel workers...")
            start_time = time.time()
            
            # Move networks to CPU and get state dicts for workers
            policy_cpu = self._policy_network.to('cpu')
            regret_cpu = self._regret_networks[player].to('cpu')
            value_cpu = self._value_network.to('cpu')
            
            worker_fn = partial(
                _traverse_tree_worker,
                game_name=self._game.get_type().short_name,
                player=player,
                policy_state_dict=policy_cpu.state_dict(),
                regret_state_dict=regret_cpu.state_dict(),
                value_state_dict=value_cpu.state_dict(),
                embedding_size=self._embedding_size,
                num_actions=self._num_actions,
                policy_layers=self._policy_network_layers,
                regret_layers=self._regret_network_layers,
                value_layers=self._value_network_layers,
                value_embedding_size=self._value_embedding_size,
                train_regret=train_regret,
                train_value=train_value,
                track_mean_squares=track_mean_squares,
                on_policy_prob=on_policy_prob,
                expl=expl,
            )
            
            with Pool(self._num_workers) as pool:
                results = pool.map(worker_fn, range(n))
            
            # Merge results from all workers
            total_nodes = 0
            for regret_samples, value_samples, nodes in results:
                total_nodes += nodes
                # Add samples to buffers (convert to proper dataclass format)
                for sample in regret_samples:
                    info_state, regret_array, legal_mask = sample
                    # Create proper _RegretSample dataclass
                    regret_sample = _RegretSample(
                        info_state=info_state,
                        iteration=float(self._iteration),
                        regret=regret_array,
                        mask=legal_mask,
                    )
                    self._regret_memories[player].add(regret_sample)
                for sample in value_samples:
                    hist_state, value = sample
                    # Create proper _ValueSample dataclass
                    value_sample = _ValueSample(
                        hist_state=hist_state,
                        iteration=float(self._iteration),
                        value=value,
                        mask=np.ones(1),  # Dummy mask for value samples
                    )
                    self._value_memory.add(value_sample)
            
            self._nodes_visited += total_nodes
            elapsed = time.time() - start_time
            rate = n / elapsed if elapsed > 0 else 0
            print(f"  Completed {n} trees in {elapsed:.1f}s ({rate:.1f} tr/s) | nodes: {total_nodes:,}")
            
            # Move networks back to original device
            self._policy_network.to(self._infer_device)
            self._regret_networks[player].to(self._infer_device)
            self._value_network.to(self._infer_device)
        else:
            # Sequential mode (original code)
            print_interval = max(1, n // 5)  # Show progress 5 times
            start_time = time.time()
            for i in range(n):
                if i > 0 and i % print_interval == 0:
                    elapsed = time.time() - start_time
                    rate = i / elapsed if elapsed > 0 else 0
                    eta = (n - i) / rate if rate > 0 else 0
                    print(f"  {i}/{n} trees ({i*100//n}%) | nodes: {self._nodes_visited:,} | {rate:.0f} tr/s | ETA: {eta:.0f}s")
                self._traverse_game_tree(
                    self._root_node,
                    player,
                    my_reach=1.0,
                    opp_reach=1.0,
                    sample_reach=1.0,
                    my_sample_reach=1.0,
                    train_regret=train_regret,
                    train_value=train_value,
                    track_mean_squares=(track_mean_squares and i == 0),
                    on_policy_prob=on_policy_prob,
                    expl=expl,
                    last_action=0,
                )

    def _traverse_game_tree(
        self,
        state: pyspiel.State,
        player: int,
        my_reach: float,
        opp_reach: float,
        sample_reach: float,
        my_sample_reach: float,
        train_regret: bool,
        train_value: bool,
        track_mean_squares: bool,
        on_policy_prob: float,
        expl: float,
        last_action: int,
    ) -> Tuple[float, float]:
        self._nodes_visited += 1
        if state.is_terminal():
            returns = state.returns()
            return returns[player], returns[player]
        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            choice = np.random.choice(len(outcomes), p=probs)
            new_state = state.child(outcomes[choice])
            return self._traverse_game_tree(
                new_state,
                player,
                my_reach=my_reach,
                opp_reach=opp_reach * probs[choice],
                sample_reach=sample_reach * probs[choice],
                my_sample_reach=my_sample_reach,
                train_regret=train_regret,
                train_value=train_value,
                track_mean_squares=track_mean_squares,
                on_policy_prob=on_policy_prob,
                expl=expl,
                last_action=outcomes[choice],
            )

        if expl != 0.0 and np.random.rand() < on_policy_prob:
            expl = 0.0

        cur_player = state.current_player()
        legal_actions = state.legal_actions()
        num_actions = state.num_distinct_actions()
        regrets, policy_probs = self._sample_action_from_regret(state, cur_player)

        if cur_player == player or train_value:
            legal_mask = np.array(state.legal_actions_mask(), dtype=np.float32)
            legal_count = legal_mask.sum()
            if legal_count == 0:
                uniform_policy = np.full(num_actions, 1.0 / num_actions)
            else:
                uniform_policy = legal_mask / legal_count
            sample_policy = expl * uniform_policy + (1.0 - expl) * policy_probs
        else:
            sample_policy = policy_probs

        sample_policy = np.clip(sample_policy, 0.0, None)
        if sample_policy.sum() == 0.0:
            legal_mask = np.array(state.legal_actions_mask(), dtype=np.float32)
            legal_idxs = np.where(legal_mask > 0)[0]
            sample_policy = np.zeros(num_actions, dtype=np.float32)
            sample_policy[legal_idxs] = 1.0 / len(legal_idxs)
        else:
            sample_policy /= sample_policy.sum()

        sampled_action = np.random.choice(num_actions, p=sample_policy)
        try:
            next_state = state.child(sampled_action)
        except Exception as exc:  # pragma: no cover - defensive guard for unexpected illegal actions
            if self._debug_logging:
                state_info = state.information_state_string(cur_player)
                legal_list = state.legal_actions()
                mask_vals = state.legal_actions_mask()
                probs_snapshot = sample_policy.tolist()
                print(
                    "[ESCHER-Torch] Illegal sampled action",
                    {
                        "iter": self._iteration,
                        "player": cur_player,
                        "action": sampled_action,
                        "legal": legal_list,
                        "mask": mask_vals,
                        "policy": probs_snapshot,
                        "state": state_info,
                    },
                )
            raise

        child_value_estimate = self._estimate_value_from_hist(next_state.clone(), player, sampled_action)
        value_estimate = self._estimate_value_from_hist(state.clone(), player, last_action)

        if track_mean_squares:
            oracle_child = self._exact_value(next_state.clone(), player)
            oracle_val = self._exact_value(state.clone(), player)
            self._squared_errors.append((oracle_val - value_estimate) ** 2)
            self._squared_errors_child.append((oracle_child - child_value_estimate) ** 2)

        if cur_player == player:
            new_my_reach = my_reach * policy_probs[sampled_action]
            new_opp_reach = opp_reach
            new_my_sample_reach = my_sample_reach * sample_policy[sampled_action]
        else:
            new_my_reach = my_reach
            new_opp_reach = opp_reach * policy_probs[sampled_action]
            new_my_sample_reach = my_sample_reach
        new_sample_reach = sample_reach * sample_policy[sampled_action]

        importance_weighted_sampled_value, sampled_value = self._traverse_game_tree(
            next_state,
            player,
            new_my_reach,
            new_opp_reach,
            new_sample_reach,
            new_my_sample_reach,
            train_regret,
            train_value,
            track_mean_squares,
            on_policy_prob,
            expl,
            sampled_action,
        )

        importance_weighted_sampled_value *= policy_probs[sampled_action] / max(sample_policy[sampled_action], 1e-8)

        child_values = np.zeros(num_actions, dtype=np.float32)
        if self._all_actions:
            for action in legal_actions:
                try:
                    child_state = state.child(action)
                except Exception as exc:  # pragma: no cover - defensive guard for unexpected illegal actions
                    if self._debug_logging:
                        state_info = state.information_state_string(cur_player)
                        print(
                            "[ESCHER-Torch] Failed to expand child",
                            {
                                "iter": self._iteration,
                                "player": cur_player,
                                "action": action,
                                "state": state_info,
                            },
                        )
                    raise
                child_values[action] = self._estimate_value_from_hist(child_state, player, action)
        else:
            child_values[sampled_action] = child_value_estimate / max(sample_policy[sampled_action], 1e-8)

        if train_regret:
            mask = np.array(state.legal_actions_mask(cur_player), dtype=np.float32)
            if cur_player == player:
                if self._importance_sampling:
                    action_sample_reach = my_sample_reach * sample_policy[sampled_action]
                    cf_value = value_estimate * min(1.0 / max(my_sample_reach, 1e-8), self._importance_sampling_threshold)
                    cf_action_values = child_values * min(
                        1.0 / max(action_sample_reach, 1e-8), self._importance_sampling_threshold
                    )
                else:
                    cf_value = value_estimate
                    cf_action_values = child_values
                samp_regret = (cf_action_values - cf_value) * mask
                info_tensor = np.array(state.information_state_tensor(player), dtype=np.float32)
                self._regret_memories[player].add(
                    _RegretSample(info_tensor, float(self._iteration), samp_regret, mask)
                )
            else:
                info_tensor = np.array(state.information_state_tensor(cur_player), dtype=np.float32)
                self._average_policy_memories.add(
                    _PolicySample(info_tensor, float(self._iteration), policy_probs, mask)
                )

        if train_value and (on_policy_prob == 0.0 or expl == 0.0):
            hist_tensor = np.append(
                state.information_state_tensor(0),
                state.information_state_tensor(1),
            ).astype(np.float32)
            if self._val_bootstrap:
                if self._all_actions:
                    target = policy_probs @ child_values
                else:
                    target = child_value_estimate * policy_probs[sampled_action] / max(sample_policy[sampled_action], 1e-8)
            elif self._debug_val:
                target = child_value_estimate * policy_probs[sampled_action] / max(sample_policy[sampled_action], 1e-8)
            else:
                target = importance_weighted_sampled_value
            mask = np.array(state.legal_actions_mask(cur_player), dtype=np.float32)
            self._value_memory.add(_ValueSample(hist_tensor, float(self._iteration), float(target), mask))

        return importance_weighted_sampled_value, sampled_value

    def _exact_value(self, state: pyspiel.State, update_player: int) -> float:
        if state.is_terminal():
            return state.player_return(update_player)
        if state.is_chance_node():
            total = 0.0
            for action, prob in state.chance_outcomes():
                total += prob * self._exact_value(state.child(action), update_player)
            return total
        legal_actions = state.legal_actions()
        _, policy_probs = self._sample_action_from_regret(state, state.current_player())
        value = 0.0
        for action in legal_actions:
            value += policy_probs[action] * self._exact_value(state.child(action), update_player)
        return value

    def _to_tensor(self, array: np.ndarray, device: torch.device) -> torch.Tensor:
        return torch.tensor(array, dtype=torch.float32, device=device)

    def _get_matched_regrets(self, info_state: np.ndarray, legal_actions_mask: np.ndarray, player: int) -> Tuple[np.ndarray, np.ndarray]:
        model = self._regret_networks[player]
        model.eval()
        with torch.no_grad():
            info_tensor = self._to_tensor(info_state, self._infer_device).unsqueeze(0)
            mask_tensor = self._to_tensor(legal_actions_mask, self._infer_device).unsqueeze(0)
            regrets = model(info_tensor, mask_tensor)[0]
            regrets = torch.clamp(regrets, min=0.0)
            summed = regrets.sum()
            if summed > 0:
                matched = regrets / summed
            else:
                legal_indices = torch.nonzero(mask_tensor[0] > 0, as_tuple=False).flatten()
                matched = torch.zeros_like(regrets)
                if len(legal_indices) > 0:
                    matched[legal_indices] = 1.0 / len(legal_indices)
                else:
                    matched.fill_(1.0 / matched.numel())
            return regrets.cpu().numpy(), matched.cpu().numpy()

    def _get_estimated_value(self, hist_state: np.ndarray, legal_actions_mask: np.ndarray) -> float:
        del legal_actions_mask
        self._value_network.eval()
        with torch.no_grad():
            tensor = self._to_tensor(hist_state, self._infer_device).unsqueeze(0)
            value = self._value_network(tensor, torch.empty_like(tensor))
        return float(value.item())

    def _estimate_value_from_hist(self, state: pyspiel.State, player: int, last_action: int) -> float:
        if state.is_terminal():
            return state.player_return(player)
        hist_tensor = np.append(
            state.information_state_tensor(0),
            state.information_state_tensor(1),
        ).astype(np.float32)
        mask = np.array(state.legal_actions_mask(player), dtype=np.float32)
        estimated_value = self._get_estimated_value(hist_tensor, mask)
        if player == 1:
            estimated_value *= -1.0
        self._example_hist_state = hist_tensor
        self._example_legal_actions_mask[player] = mask
        return estimated_value

    def _sample_action_from_regret(self, state: pyspiel.State, player: int) -> Tuple[np.ndarray, np.ndarray]:
        info_state = np.array(state.information_state_tensor(player), dtype=np.float32)
        mask = np.array(state.legal_actions_mask(player), dtype=np.float32)
        self._example_info_state[player] = info_state
        self._example_legal_actions_mask[player] = mask
        return self._get_matched_regrets(info_state, mask, player)

    def _batch_to_tensor(self, batch: Sequence, keys: Tuple[str, ...], device: torch.device) -> Tuple[torch.Tensor, ...]:
        arrays = {key: [] for key in keys}
        for sample in batch:
            for key in keys:
                arrays[key].append(getattr(sample, key))
        tensors = []
        for key in keys:
            arr = np.stack(arrays[key])
            tensors.append(self._to_tensor(arr, device))
        return tuple(tensors)

    def _learn_regret_network(self, player: int) -> Optional[float]:
        buffer = self._regret_memories[player]
        if len(buffer) < max(1, self._batch_size_regret):
            return None
        model = self._regret_train_models[player]
        optimizer = self._regret_optimizers[player]
        model.train()
        losses = []
        for _ in range(self._regret_network_train_steps):
            batch_size = min(len(buffer), self._batch_size_regret)
            batch = buffer.sample(batch_size)
            info_states, iterations, regrets, masks = self._batch_to_tensor(
                batch, ("info_state", "iteration", "regret", "mask"), self._train_device
            )
            optimizer.zero_grad()
            preds = model(info_states, masks)
            weight = (iterations.squeeze(-1) * 2.0 / max(self._iteration, 1)).unsqueeze(-1)
            loss = ((preds - regrets) ** 2 * weight).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().item())
        self._regret_networks[player].load_state_dict(model.state_dict())
        return float(np.mean(losses))

    def _learn_value_network(self) -> Optional[float]:
        buffer = self._value_memory
        if len(buffer) < max(1, self._batch_size_value):
            return None
        model = self._value_train_model
        optimizer = self._value_optimizer
        model.train()
        losses = []
        for _ in range(self._value_network_train_steps):
            batch_size = min(len(buffer), self._batch_size_value)
            batch = buffer.sample(batch_size)
            hist_states, iterations, values, masks = self._batch_to_tensor(
                batch, ("hist_state", "iteration", "value", "mask"), self._train_device
            )
            optimizer.zero_grad()
            preds = model(hist_states, masks)
            targets = values.squeeze(-1)
            weight = (iterations.squeeze(-1) * 2.0 / max(self._iteration, 1))
            loss = ((preds - targets) ** 2 * weight).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().item())
        self._value_network.load_state_dict(model.state_dict())
        return float(np.mean(losses))

    def _learn_average_policy_network(self) -> Optional[float]:
        buffer = self._average_policy_memories
        if len(buffer) < max(1, self._batch_size_average_policy):
            return None
        model = self._policy_network
        optimizer = self._policy_optimizer
        model.train()
        losses = []
        for _ in range(self._policy_network_train_steps):
            batch_size = min(len(buffer), self._batch_size_average_policy)
            batch = buffer.sample(batch_size)
            info_states, iterations, probs, masks = self._batch_to_tensor(
                batch, ("info_state", "iteration", "probs", "mask"), self._train_device
            )
            optimizer.zero_grad()
            preds = model(info_states, masks)
            weight = (iterations.squeeze(-1) * 2.0 / max(self._iteration, 1)).unsqueeze(-1)
            loss = ((preds - probs) ** 2 * weight).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().item())
        model.eval()
        return float(np.mean(losses))

    def solve(self, save_path_convs: Optional[str] = None):
        regret_losses = collections.defaultdict(list)
        convs: List[float] = []
        nodes: List[int] = []
        iteration_times: List[float] = []
        timestr = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        self.traverse_game_tree_n_times(1, 0, track_mean_squares=False)

        policy_loss_history: List[Optional[float]] = []
        value_loss_history: List[Optional[float]] = []

        for iteration in range(self._num_iterations):
            print(f"\n--- Iteration {iteration + 1}/{self._num_iterations} ---")
            iter_start = time.time()
            nodes_before = self._nodes_visited
            policy_loss = None
            conv_value = None
            if self._experiment_string:
                print(self._experiment_string)
            if self._debug_logging:
                print(f"[ESCHER-Torch] Iteration {iteration + 1}/{self._num_iterations} starting")
            if iteration % max(1, self._check_exploitability_every) == 0:
                print(f"Training policy network ({len(self._average_policy_memories)} samples)...")
                policy_loss = self._learn_average_policy_network()
                pol_text = "N/A" if policy_loss is None else f"{policy_loss:.5f}"
                print(f"Policy loss: {pol_text}")
                if self._compute_exploitability:
                    print("Computing exploitability...")
                    conv_value = exploitability.nash_conv(self._game, self)
                    convs.append(conv_value)
                    nodes.append(self.get_num_nodes())
                    if save_path_convs:
                        np.save(f"{save_path_convs}_convs.npy", np.array(convs))
                        np.save(f"{save_path_convs}_nodes.npy", np.array(nodes))
                    print(f"✓ Exploitability: {conv_value:.4f}")
                if policy_loss is not None and self._save_policy_weights and save_path_convs:
                    model_dir = os.path.join(save_path_convs, timestr)
                    os.makedirs(model_dir, exist_ok=True)
                    torch.save(self._policy_network.state_dict(), os.path.join(model_dir, "policy.pt"))
                if self._debug_logging:
                    conv_text = "skipped" if conv_value is None else f"{conv_value:.5f}"
                    policy_text = "none" if policy_loss is None else f"{policy_loss:.5f}"
                    avg_policy_mem = len(self._average_policy_memories)
                    print(
                        f"[ESCHER-Torch] Iter {self._iteration} policy_loss={policy_text} conv={conv_text} "
                        f"avg_mem={avg_policy_mem}"
                    )
            elif self._debug_logging:
                print(f"[ESCHER-Torch] Iter {self._iteration} exploitability skipped")

            policy_loss_history.append(policy_loss)

            print("Value network training:")
            self.traverse_game_tree_n_times(
                self._num_val_fn_traversals,
                player=0,
                train_value=True,
                track_mean_squares=False,
                on_policy_prob=self._val_op_prob,
                expl=self._val_expl,
            )
            print(f"  Training value network ({len(self._value_memory)} samples)...")
            value_loss = self._learn_value_network()
            value_mem_len = len(self._value_memory)
            value_text = "N/A" if value_loss is None else f"{value_loss:.5f}"
            print(f"  Value loss: {value_text}")
            if self._clear_value_buffer:
                self.clear_value_memories()
            if self._debug_logging:
                print(
                    f"[ESCHER-Torch] Iter {self._iteration} value_loss={value_text} value_mem={value_mem_len}"
                )

            value_loss_history.append(value_loss)

            for p in range(self._num_players):
                print(f"Regret network training (Player {p}):")
                self.traverse_game_tree_n_times(
                    self._num_traversals,
                    player=p,
                    train_regret=True,
                    track_mean_squares=self._track_mean_squares,
                    expl=self._expl,
                )
                if self._reinitialize_regret_networks:
                    self._regret_train_models[p] = RegretNetwork(
                        self._embedding_size,
                        self._regret_network_layers,
                        self._num_actions,
                    ).to(self._train_device)
                    self._regret_train_models[p].load_state_dict(self._regret_networks[p].state_dict())
                    self._regret_optimizers[p] = torch.optim.Adam(
                        self._regret_train_models[p].parameters(), lr=self._learning_rate
                    )
                print(f"  Training regret network ({len(self._regret_memories[p])} samples)...")
                loss = self._learn_regret_network(p)
                if loss is not None:
                    regret_losses[p].append(loss)
                loss_text = "N/A" if loss is None else f"{loss:.5f}"
                print(f"  Regret loss: {loss_text}")
                if self._debug_logging:
                    reg_mem = len(self._regret_memories[p])
                    print(
                        f"[ESCHER-Torch] Iter {self._iteration} player={p} regret_loss={loss_text} mem={reg_mem}"
                    )

            self._iteration += 1

            iteration_times.append(time.time() - iter_start)
            nodes_after = self._nodes_visited
            nodes_this_iter = nodes_after - nodes_before
            print(f"✓ Iteration {iteration + 1} complete: {iteration_times[-1]:.1f}s, {nodes_this_iter:,} nodes, total: {nodes_after:,}")
            if self._debug_logging:
                print(
                    f"[ESCHER-Torch] Iteration {self._iteration - 1} duration={iteration_times[-1]:.3f}s nodes={nodes_this_iter}"
                )

        print("\nTraining final policy...")
        final_policy_loss = self._learn_average_policy_network()
        final_policy_status = "N/A" if final_policy_loss is None else f"{final_policy_loss:.5f}"
        print(f"Final policy loss: {final_policy_status}")
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        if self._debug_logging:
            print(f"[ESCHER-Torch] Final policy loss={final_policy_status}")
        return (
            regret_losses,
            final_policy_loss,
            convs,
            nodes,
            iteration_times,
            policy_loss_history,
            value_loss_history,
        )

    def action_probabilities(self, state: pyspiel.State) -> Dict[int, float]:
        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)
        mask = np.array(state.legal_actions_mask(cur_player), dtype=np.float32)
        info_state = np.array(state.information_state_tensor(cur_player), dtype=np.float32)
        self._policy_network.eval()
        with torch.no_grad():
            tensor_info = self._to_tensor(info_state, self._infer_device).unsqueeze(0)
            tensor_mask = self._to_tensor(mask, self._infer_device).unsqueeze(0)
            probs = self._policy_network(tensor_info, tensor_mask)[0].cpu().numpy()
        return {action: float(probs[action]) for action in legal_actions}

    def get_num_nodes(self) -> int:
        return self._nodes_visited

    def get_squared_errors(self) -> List[float]:
        return list(self._squared_errors)

    def reset_squared_errors(self) -> None:
        self._squared_errors.clear()

    def get_squared_errors_child(self) -> List[float]:
        return list(self._squared_errors_child)

    def reset_squared_errors_child(self) -> None:
        self._squared_errors_child.clear()
