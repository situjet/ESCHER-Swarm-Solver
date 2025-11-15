"""Path planning utilities for the large Swarm-AD OpenSpiel game."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

Point = Tuple[float, float]


@dataclass(frozen=True)
class Bounds:
    row_min: float
    row_max: float
    col_min: float
    col_max: float

    def clamp(self, point: Point) -> Point:
        row = min(max(point[0], self.row_min), self.row_max)
        col = min(max(point[1], self.col_min), self.col_max)
        return row, col


@dataclass(frozen=True)
class CircleObstacle:
    center: Point
    radius: float
    padding: float = 0.0

    @property
    def effective_radius(self) -> float:
        return max(0.0, self.radius - self.padding)


def _segment_circle_intersects(a: Point, b: Point, center: Point, radius: float) -> bool:
    if radius <= 0:
        return False
    ax, ay = a
    bx, by = b
    cx, cy = center
    abx = bx - ax
    aby = by - ay
    acx = cx - ax
    acy = cy - ay
    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq == 0:
        return math.dist(a, center) <= radius
    t = (acx * abx + acy * aby) / ab_len_sq
    t = max(0.0, min(1.0, t))
    closest = (ax + abx * t, ay + aby * t)
    return math.dist(closest, center) <= radius


def segment_clear(a: Point, b: Point, obstacles: Sequence[CircleObstacle]) -> bool:
    return all(not _segment_circle_intersects(a, b, obs.center, obs.effective_radius) for obs in obstacles)


def _biased_sample(
    goal: Point,
    bounds: Bounds,
    rng: random.Random,
    *,
    goal_bias: float,
    bias_points: Optional[Sequence[Point]] = None,
) -> Point:
    if rng.random() < goal_bias:
        return goal
    if bias_points:
        choice = rng.choice(bias_points)
        jitter = (rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0))
        return bounds.clamp((choice[0] + jitter[0], choice[1] + jitter[1]))
    row = rng.uniform(bounds.row_min, bounds.row_max)
    col = rng.uniform(bounds.col_min, bounds.col_max)
    return row, col


def _nearest(points: Sequence[Point], sample: Point) -> int:
    best_idx = 0
    best_dist = float("inf")
    for idx, point in enumerate(points):
        dist = math.dist(point, sample)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx


def _step_towards(origin: Point, target: Point, step_size: float, bounds: Bounds) -> Point:
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    distance = math.dist(origin, target)
    if distance <= step_size:
        return bounds.clamp(target)
    scale = step_size / distance
    row = origin[0] + (target[0] - origin[0]) * scale
    col = origin[1] + (target[1] - origin[1]) * scale
    return bounds.clamp((row, col))


def rrt_path(
    start: Point,
    goal: Point,
    obstacles: Sequence[CircleObstacle],
    bounds: Bounds,
    *,
    rng: Optional[random.Random] = None,
    step_size: float = 1.4,
    goal_sample_prob: float = 0.2,
    max_iters: int = 1200,
    bias_points: Optional[Sequence[Point]] = None,
) -> List[Point]:
    """Computes an obstacle-aware RRT path."""
    if rng is None:
        rng = random.Random()
    nodes: List[Point] = [bounds.clamp(start)]
    parents: List[int] = [-1]
    for _ in range(max_iters):
        sample = _biased_sample(goal, bounds, rng, goal_bias=goal_sample_prob, bias_points=bias_points)
        nearest_idx = _nearest(nodes, sample)
        candidate = _step_towards(nodes[nearest_idx], sample, step_size, bounds)
        if not segment_clear(nodes[nearest_idx], candidate, obstacles):
            continue
        nodes.append(candidate)
        parents.append(nearest_idx)
        if segment_clear(candidate, goal, obstacles):
            nodes.append(goal)
            parents.append(len(nodes) - 2)
            break
    if len(nodes) == 1:
        return [start, goal]
    path = []
    idx = len(nodes) - 1
    while idx >= 0:
        path.append(nodes[idx])
        idx = parents[idx]
        if idx == -1:
            break
    path.reverse()
    if not path or path[0] != start:
        path.insert(0, start)
    if path[-1] != goal:
        path.append(goal)
    return path


def smooth_path(
    path: Sequence[Point],
    obstacles: Sequence[CircleObstacle],
    *,
    rng: Optional[random.Random] = None,
    attempts: int = 64,
) -> List[Point]:
    if rng is None:
        rng = random.Random()
    if len(path) < 3:
        return list(path)
    points = list(path)
    for _ in range(attempts):
        if len(points) < 3:
            break
        i = rng.randint(0, len(points) - 3)
        j = rng.randint(i + 2, len(points) - 1)
        if segment_clear(points[i], points[j], obstacles):
            points[i + 1 : j] = []
    return points


def path_length(path: Sequence[Point]) -> float:
    return sum(math.dist(path[i], path[i + 1]) for i in range(len(path) - 1))


def sample_path(
    path: Sequence[Point],
    *,
    step: float = 0.4,
) -> List[Tuple[float, float, float]]:
    """Returns (row, col, distance_from_start) samples along the path."""
    if not path:
        return []
    samples: List[Tuple[float, float, float]] = []
    cumulative = 0.0
    samples.append((path[0][0], path[0][1], cumulative))
    for start, end in zip(path, path[1:]):
        segment = math.dist(start, end)
        if segment == 0:
            continue
        steps = max(1, int(segment / step))
        for step_idx in range(1, steps + 1):
            t = step_idx / steps
            point = (
                start[0] + (end[0] - start[0]) * t,
                start[1] + (end[1] - start[1]) * t,
            )
            cumulative += segment / steps
            samples.append((point[0], point[1], cumulative))
    if samples[-1][:2] != path[-1]:
        samples.append((path[-1][0], path[-1][1], cumulative))
    return samples


__all__ = [
    "Bounds",
    "CircleObstacle",
    "rrt_path",
    "smooth_path",
    "segment_clear",
    "sample_path",
    "path_length",
]
