"""Blueprint helpers for mid-waypoint pathfinding."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

Point = Tuple[float, float]

MIDPOINT_STRATEGIES: Tuple[str, ...] = (
    "direct",
    "fan_left",
    "fan_right",
    "loiter",
)


@dataclass
class BlueprintContext:
    entry: Point
    destination: Point
    tot_delay: float
    arena_width: float
    arena_height: float
    ad_positions: Sequence[Point]


def _offset(point: Point, angle: float, distance: float) -> Point:
    return point[0] + math.sin(angle) * distance, point[1] + math.cos(angle) * distance


def _angle_between(a: Point, b: Point) -> float:
    return math.atan2(b[0] - a[0], b[1] - a[1])


def _clamp(point: Point, ctx: BlueprintContext) -> Point:
    return (
        min(max(point[0], 0.0), ctx.arena_height),
        min(max(point[1], 0.0), ctx.arena_width),
    )


def _lerp(a: Point, b: Point, t: float) -> Point:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _noisy(point: Point, ctx: BlueprintContext, rng: random.Random, scale: float) -> Point:
    spread_row = ctx.arena_height * scale
    spread_col = ctx.arena_width * scale
    noisy_point = (
        point[0] + rng.uniform(-spread_row, spread_row),
        point[1] + rng.uniform(-spread_col, spread_col),
    )
    return _clamp(noisy_point, ctx)


def _random_far_point(entry: Point, ctx: BlueprintContext, rng: random.Random) -> Point:
    return (
        rng.uniform(ctx.arena_height * 0.35, ctx.arena_height),
        rng.uniform(0.0, ctx.arena_width),
    )


def blueprint_midpoints(strategy: str, ctx: BlueprintContext, rng: random.Random) -> List[Point]:
    """Returns two stochastic mid-waypoints for the requested strategy."""
    strategy = strategy.lower()
    entry = ctx.entry
    destination = ctx.destination
    heading = _angle_between(entry, destination)
    lateral = ctx.arena_width * 0.25
    depth = ctx.arena_height * 0.35

    points: List[Point]

    if strategy == "direct":
        first = _noisy(_lerp(entry, destination, 0.33), ctx, rng, 0.2)
        second = _noisy(_lerp(entry, destination, 0.66), ctx, rng, 0.2)
        points = [first, second]
    elif strategy == "fan_left":
        mid1 = _offset(entry, heading + math.pi / 2, lateral)
        mid2 = _offset(destination, heading + math.pi / 2, depth)
        points = [_noisy(mid1, ctx, rng, 0.15), _noisy(mid2, ctx, rng, 0.2)]
    elif strategy == "fan_right":
        mid1 = _offset(entry, heading - math.pi / 2, lateral)
        mid2 = _offset(destination, heading - math.pi / 2, depth)
        points = [_noisy(mid1, ctx, rng, 0.15), _noisy(mid2, ctx, rng, 0.2)]
    elif strategy == "loiter":
        loop_start = _offset(destination, heading + math.pi / 2, depth * 0.4)
        loop_end = _offset(destination, heading - math.pi / 2, depth * 0.4)
        points = [_noisy(loop_start, ctx, rng, 0.25), _noisy(loop_end, ctx, rng, 0.25)]
    else:
        points = [_random_far_point(entry, ctx, rng), _random_far_point(entry, ctx, rng)]

    return points


__all__ = ["MIDPOINT_STRATEGIES", "BlueprintContext", "blueprint_midpoints"]
