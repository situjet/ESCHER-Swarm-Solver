"""Tests for reward accounting in the Swarm Defense OpenSpiel game."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarm_defense_game import (  # noqa: E402
    AD_RESOLVE_ACTION_BASE,
    DronePlan,
    DroneTargetStrike,
    Phase,
    SwarmDefenseGame,
    TargetCluster,
)


def test_target_kill_reward_only_once() -> None:
    """Even if multiple drones succeed, a target grants damage only once."""

    game = SwarmDefenseGame()
    state = game.new_initial_state()

    # Configure three targets matching the main game's expectations.
    state._targets = [
        TargetCluster(row=8, col=0, value=10.0),
        TargetCluster(row=9, col=1, value=20.0),
        TargetCluster(row=10, col=2, value=40.0),
    ]
    state._target_destroyed = [False] * len(state._targets)

    # Two drones both attempt to strike target 0 successfully.
    state._drone_plans = [
        DronePlan(entry_row=0, entry_col=0, target_idx=0, tot_idx=0),
        DronePlan(entry_row=0, entry_col=1, target_idx=0, tot_idx=0),
    ]
    state._pending_target_strikes = [
        DroneTargetStrike(drone_idx=0, target_idx=0, probability=1.0),
        DroneTargetStrike(drone_idx=1, target_idx=0, probability=1.0),
    ]
    state._next_target_strike_index = 0
    state._damage_from_targets = 0.0
    state._phase = Phase.TARGET_DAMAGE_RESOLUTION

    success_action = AD_RESOLVE_ACTION_BASE + 1

    # First success destroys the target and awards damage.
    state._apply_target_damage_resolution(success_action)
    assert state._damage_from_targets == state._targets[0].value
    assert state._target_destroyed[0] is True
    assert state._drone_plans[0].damage_inflicted == state._targets[0].value
    assert state._phase == Phase.TARGET_DAMAGE_RESOLUTION

    # Second success against the already-destroyed target gives no extra reward.
    state._apply_target_damage_resolution(success_action)
    assert state._damage_from_targets == state._targets[0].value
    assert state._drone_plans[1].damage_inflicted == 0.0
    assert state.returns() == [state._targets[0].value, -state._targets[0].value]
