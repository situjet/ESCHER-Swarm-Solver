"""Unit tests covering interceptor/AD timing constraints."""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GAME_DIR = ROOT / "Swarm-AD-OpenSpiel"
sys.path.insert(0, str(GAME_DIR))

from swarm_defense_game import (  # noqa: E402
    DronePlan,
    _arrival_time_to_point,
    _compute_interceptor_intercept,
    _path_exposure_stats,
)
import demo_visualizer  # type: ignore  # noqa: E402


def test_interceptor_cannot_hit_after_target_contact() -> None:
    """Interceptors launched from row 15 cannot overtake extremely short attacks."""
    destination = (1, 0)
    plan = DronePlan(entry_row=0, entry_col=0, target_idx=0, tot_idx=0)
    intercept = _compute_interceptor_intercept(plan, destination)
    assert intercept is None, "intercept should be impossible once the drone is effectively on target"


def test_interceptor_hit_happens_before_target() -> None:
    """When geometry allows, intercepts are scheduled strictly before target impact."""
    destination = (15, 0)
    plan = DronePlan(entry_row=0, entry_col=0, target_idx=0, tot_idx=0)
    intercept = _compute_interceptor_intercept(plan, destination)
    arrival = _arrival_time_to_point(plan, destination)
    assert intercept is not None
    intercept_time, point = intercept
    assert intercept_time < arrival
    assert 0.0 <= point[0] <= destination[0]
    assert math.isclose(point[1], 0.0, abs_tol=1e-6)


def test_ad_entry_distance_tracks_first_contact() -> None:
    """AD exposure helper returns a finite entry distance for downstream timing checks."""
    ad_position = (8, 0)
    entry = (0, 0)
    destination = (15, 0)
    exposure, entry_point, entry_distance = _path_exposure_stats(ad_position, entry, destination)
    assert exposure > 0.0
    assert entry_point is not None
    assert entry_distance is not None
    total_distance = math.dist(entry, destination)
    assert 0.0 <= entry_distance < total_distance


def test_snapshot_contains_ad_kill_coordinates() -> None:
    """Deterministic seed should produce drones with logged AD intercept metadata."""
    state, _ = demo_visualizer.play_episode(1005)
    snapshot = state.snapshot()
    kills = [
        drone
        for drone in snapshot["drones"]
        if isinstance(drone.get("destroyed_by"), str) and drone["destroyed_by"].startswith("ad:")
    ]
    assert kills, "expected at least one AD kill for visualization regression"
    for drone in kills:
        assert drone["intercepts"], "AD kill should store hit metadata"
        for ad_idx, point, tstamp in drone["intercepts"]:
            assert isinstance(ad_idx, int)
            assert isinstance(point, tuple) and len(point) == 2
            assert all(isinstance(coord, float) for coord in point)
            assert tstamp > 0.0
