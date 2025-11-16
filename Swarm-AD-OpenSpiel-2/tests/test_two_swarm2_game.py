"""Unit tests for the 2Swarm2 OpenSpiel game."""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import pathlib
import sys

import pyspiel
import pytest

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from Swarm_AD_OpenSpiel_2 import two_swarm2_game as mod

# Convenience aliases
TARGET_POSITIONS = mod.TARGET_CANDIDATE_CELLS
AD_POSITIONS = mod.AD_POSITION_CANDIDATES
NUM_TARGETS = mod.NUM_TARGETS


def _resolve_all_chance(state: pyspiel.State) -> None:
    while state.current_player() == pyspiel.PlayerId.CHANCE:
        outcomes = state.chance_outcomes()
        assert outcomes, "Chance node missing outcomes"
        state.apply_action(outcomes[0][0])


def _apply_target_grid(state: pyspiel.State, cells: Sequence[Tuple[int, int]]) -> None:
    assert len(cells) == NUM_TARGETS
    for cell in cells:
        idx = TARGET_POSITIONS.index(cell)
        state.apply_action(mod.TARGET_POSITION_ACTION_BASE + idx)


def _place_ads(state: pyspiel.State, cells: Iterable[Tuple[int, int]]) -> None:
    for cell in cells:
        idx = AD_POSITIONS.index(cell)
        state.apply_action(mod.AD_ACTION_BASE + idx)


def _drone_action(entry_col: int, target_idx: int, tot_idx: int = 0) -> int:
    per_entry = mod.DRONE_TARGET_SLOTS * len(mod.TOT_CHOICES)
    entry_offset = entry_col * per_entry
    target_offset = target_idx * len(mod.TOT_CHOICES)
    return mod.DRONE_ACTION_BASE + entry_offset + target_offset + tot_idx


def _setup_default_layout(state: pyspiel.State) -> None:
    first_targets = TARGET_POSITIONS[:NUM_TARGETS]
    _apply_target_grid(state, first_targets)
    state.apply_action(mod.TARGET_VALUE_ACTION_BASE)  # deterministic permutation
    first_ads = AD_POSITIONS[: mod.NUM_AD_UNITS]
    _place_ads(state, first_ads)


def test_wave_two_unlocks_ad_targets() -> None:
    game = mod.TwoSwarm2Game()
    state = game.new_initial_state()

    _setup_default_layout(state)

    # Attempting to strike an AD during wave 1 should be illegal.
    target_idx = len(state.snapshot()["targets"]) + 0  # first AD slot
    illegal_action = _drone_action(entry_col=0, target_idx=target_idx)
    with pytest.raises(ValueError):
        state.apply_action(illegal_action)

    # Legitimate drone targeting the co-located target guarantees AD discovery.
    legal_action = _drone_action(entry_col=0, target_idx=0)
    state.apply_action(legal_action)
    state.apply_action(mod.DRONE_END_ACTION)

    state.apply_action(mod.INTERCEPT_END_ACTION)
    _resolve_all_chance(state)

    snapshot = state.snapshot()
    assert snapshot["current_wave"] == 2
    assert 0 in snapshot["discovered_ads"], "Wave one should reveal the co-located AD"

    ad_attack_action = _drone_action(entry_col=0, target_idx=len(snapshot["targets"]) + 0)
    assert ad_attack_action in state.legal_actions(), "Discovered AD must be targetable in wave two"


def test_interceptor_bonus_applied_when_unused() -> None:
    game = mod.TwoSwarm2Game()
    state = game.new_initial_state()

    _setup_default_layout(state)

    # Wave one: single drone, immediately end wave.
    state.apply_action(_drone_action(entry_col=0, target_idx=0))
    state.apply_action(mod.DRONE_END_ACTION)
    state.apply_action(mod.INTERCEPT_END_ACTION)
    _resolve_all_chance(state)

    # Wave two: deploy remaining drones quickly.
    remaining = state.snapshot()["remaining_drones"]
    for idx in range(remaining):
        entry = idx % mod.GRID_SIZE
        target = idx % NUM_TARGETS
        state.apply_action(_drone_action(entry_col=entry, target_idx=target))
    state.apply_action(mod.INTERCEPT_END_ACTION)
    _resolve_all_chance(state)

    assert state.is_terminal()
    expected_bonus = mod.NUM_INTERCEPTORS * mod.INTERCEPTOR_LEFTOVER_VALUE
    assert pytest.approx(state.returns()[0]) == -expected_bonus
    assert pytest.approx(state.returns()[1]) == expected_bonus
