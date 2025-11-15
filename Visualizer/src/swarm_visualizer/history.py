"""Game history representation used to feed the PyTak visualizer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .entities import DroneStatus


@dataclass
class DroneState:
    drone_id: str
    x: int
    y: int
    status: DroneStatus
    target_cluster: str
    tot_offset: int
    time_on_target: Optional[int]
    payload_value: int


@dataclass
class InterceptorState:
    interceptor_id: str
    x: int
    y: int
    assigned_drone: Optional[str]


@dataclass
class ADState:
    ad_id: str
    x: int
    y: int
    coverage: int
    engaged_drones: List[str]


@dataclass
class TargetState:
    cluster_id: str
    x: int
    y: int
    value: int
    tot_offset: int
    is_destroyed: bool


@dataclass
class GameStateSnapshot:
    tick: int
    timestamp: float
    drones: List[DroneState]
    interceptors: List[InterceptorState]
    ad_units: List[ADState]
    targets: List[TargetState]


@dataclass
class GameHistory:
    scenario_id: str
    metadata: Dict[str, str]
    snapshots: List[GameStateSnapshot] = field(default_factory=list)

    def add_snapshot(self, snapshot: GameStateSnapshot) -> None:
        self.snapshots.append(snapshot)

    def latest(self) -> GameStateSnapshot:
        if not self.snapshots:
            raise ValueError("No snapshots in history")
        return self.snapshots[-1]
