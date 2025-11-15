"""Time-step simulator that produces a game history."""

from __future__ import annotations

import time
from random import Random
from typing import List

from .config import ScenarioBundle
from .entities import Drone, DroneStatus, GridPosition, Interceptor, TargetCluster
from .history import ADState, GameHistory, GameStateSnapshot, InterceptorState, TargetState, DroneState
from .scenario import SwarmScenario


class SwarmGameSimulator:
    def __init__(self, scenario: SwarmScenario, bundle: ScenarioBundle | None = None):
        self.scenario = scenario
        self.bundle = bundle or scenario.bundle
        # Offset random seed to avoid identical layouts & AD rolls.
        seed = (self.bundle.scenario.seed or int(time.time())) + 31
        self._rng = Random(seed)
        self._current_time = 0

    def run(self, max_ticks: int = 35) -> GameHistory:
        history = GameHistory(
            scenario_id=self.scenario.scenario_id,
            metadata={
                "grid_size": str(self.bundle.grid.size),
                "num_attackers": str(self.bundle.scenario.num_attackers),
                "num_interceptors": str(self.bundle.scenario.num_interceptors),
                "ad_probability": str(self.bundle.scenario.ad_kill_probability),
            },
        )

        for tick in range(max_ticks):
            snapshot = self._capture_snapshot(tick)
            history.add_snapshot(snapshot)
            if self._all_attackers_resolved():
                break
            self._advance_state()
        return history

    # ------------------------------------------------------------------
    def _capture_snapshot(self, tick: int) -> GameStateSnapshot:
        timestamp = float(self._current_time)
        drones = [
            DroneState(
                drone_id=drone.drone_id,
                x=drone.position.x,
                y=drone.position.y,
                status=drone.status,
                target_cluster=drone.target_cluster,
                tot_offset=drone.tot_offset,
                time_on_target=drone.time_on_target,
                payload_value=drone.payload_value,
            )
            for drone in self.scenario.attackers
        ]

        interceptors = [
            InterceptorState(
                interceptor_id=interceptor.interceptor_id,
                x=interceptor.position.x,
                y=interceptor.position.y,
                assigned_drone=interceptor.assigned_drone,
            )
            for interceptor in self.scenario.interceptors
        ]

        ad_units = [
            ADState(
                ad_id=ad.ad_id,
                x=ad.position.x,
                y=ad.position.y,
                coverage=ad.coverage,
                engaged_drones=sorted(ad.engaged_drones),
            )
            for ad in self.scenario.ad_units
        ]

        targets = [
            TargetState(
                cluster_id=target.cluster_id,
                x=target.position.x,
                y=target.position.y,
                value=target.value,
                tot_offset=target.tot_offset,
                is_destroyed=target.is_destroyed,
            )
            for target in self.scenario.targets
        ]

        return GameStateSnapshot(
            tick=tick,
            timestamp=timestamp,
            drones=drones,
            interceptors=interceptors,
            ad_units=ad_units,
            targets=targets,
        )

    def _advance_state(self) -> None:
        self._current_time += 1
        self._advance_attackers()
        self._advance_interceptors()

    def _advance_attackers(self) -> None:
        grid_size = self.bundle.grid.size
        target_lookup = {target.cluster_id: target for target in self.scenario.targets}
        for drone in self.scenario.attackers:
            if drone.status is not DroneStatus.ACTIVE:
                continue
            target = target_lookup[drone.target_cluster]
            drone.next_position = drone.position.step_towards(target.position).clamp(grid_size)
            drone.advance()
            if target.contains(drone.position):
                drone.mark_completed(self._current_time + drone.tot_offset)
                target.is_destroyed = True
                continue
            self._apply_ad_threats(drone)

    def _apply_ad_threats(self, drone: Drone) -> None:
        if drone.status is not DroneStatus.ACTIVE:
            return
        for ad in self.scenario.ad_units:
            if not ad.in_envelope(drone.position):
                continue
            ad.mark_engagement(drone.drone_id)
            if self._rng.random() <= self.bundle.scenario.ad_kill_probability:
                drone.mark_destroyed()
                break

    def _advance_interceptors(self) -> None:
        assignments = self._rank_attackers_by_priority()
        for interceptor, drone in zip(self.scenario.interceptors, assignments, strict=False):
            if drone is None:
                interceptor.assigned_drone = None
                continue
            interceptor.assigned_drone = drone.drone_id
            interceptor.advance(drone.position)
            if interceptor.position == drone.position and drone.status is DroneStatus.ACTIVE:
                drone.mark_destroyed()

    def _rank_attackers_by_priority(self) -> List[Drone | None]:
        active_drones = [d for d in self.scenario.attackers if d.status is DroneStatus.ACTIVE]
        active_drones.sort(key=lambda d: (-d.payload_value, d.tot_offset, d.drone_id))
        ranked: List[Drone | None] = []
        for idx in range(len(self.scenario.interceptors)):
            drone = active_drones[idx] if idx < len(active_drones) else None
            ranked.append(drone)
        return ranked

    def _all_attackers_resolved(self) -> bool:
        return all(drone.status is not DroneStatus.ACTIVE for drone in self.scenario.attackers)
