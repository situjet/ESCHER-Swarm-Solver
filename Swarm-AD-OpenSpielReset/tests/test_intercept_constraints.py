"""Unit tests covering interceptor/AD timing constraints."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarm_defense_game import (
    DronePlan,
    _arrival_time_to_point,
    _compute_interceptor_intercept,
    _path_exposure_stats,
)


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
    # Entry distance should be within the overall travel budget so AD kills stay pre-impact.
    total_distance = math.dist(entry, destination)
    assert 0.0 <= entry_distance < total_distance
