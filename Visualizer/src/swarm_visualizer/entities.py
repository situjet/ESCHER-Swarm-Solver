"""Core entities used by the swarm scenario."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


@dataclass(frozen=True)
class GridPosition:
    x: int
    y: int

    def distance_to(self, other: "GridPosition") -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)

    def step_towards(self, other: "GridPosition") -> "GridPosition":
        dx = 0
        dy = 0
        if self.x < other.x:
            dx = 1
        elif self.x > other.x:
            dx = -1
        if self.y < other.y:
            dy = 1
        elif self.y > other.y:
            dy = -1
        return GridPosition(self.x + dx, self.y + dy)

    def clamp(self, grid_size: int) -> "GridPosition":
        return GridPosition(
            max(0, min(grid_size - 1, self.x)),
            max(0, min(grid_size - 1, self.y)),
        )


class DroneStatus(str, Enum):
    ACTIVE = "active"
    DESTROYED = "destroyed"
    COMPLETED = "completed"


@dataclass
class TargetCluster:
    cluster_id: str
    position: GridPosition
    value: int
    tot_offset: int
    radius: int = 1
    is_destroyed: bool = False

    def contains(self, position: GridPosition) -> bool:
        return self.position.distance_to(position) <= self.radius


@dataclass
class Drone:
    drone_id: str
    position: GridPosition
    target_cluster: str
    next_position: GridPosition
    status: DroneStatus = DroneStatus.ACTIVE
    tot_offset: int = 0
    time_on_target: Optional[int] = None
    payload_value: int = 1
    kill_position: Optional[GridPosition] = None
    kill_tick: Optional[int] = None
    kill_source: Optional[str] = None

    def advance(self) -> None:
        if self.status is DroneStatus.ACTIVE:
            self.position = self.next_position

    def mark_destroyed(
        self,
        *,
        current_time: Optional[int] = None,
        source: Optional[str] = None,
    ) -> None:
        self.status = DroneStatus.DESTROYED
        self.kill_position = self.position
        if current_time is not None:
            self.kill_tick = current_time
        if source:
            self.kill_source = source

    def mark_completed(self, current_time: int) -> None:
        self.status = DroneStatus.COMPLETED
        self.time_on_target = current_time


@dataclass
class Interceptor:
    interceptor_id: str
    position: GridPosition
    assigned_drone: Optional[str] = None

    def advance(self, target_position: GridPosition) -> None:
        self.position = self.position.step_towards(target_position)


@dataclass
class ADUnit:
    ad_id: str
    position: GridPosition
    coverage: int = 2
    engaged_drones: set[str] = field(default_factory=set)

    def in_envelope(self, position: GridPosition) -> bool:
        return self.position.distance_to(position) <= self.coverage

    def mark_engagement(self, drone_id: str) -> None:
        self.engaged_drones.add(drone_id)
