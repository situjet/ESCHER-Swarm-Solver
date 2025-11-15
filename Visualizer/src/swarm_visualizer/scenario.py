"""Scenario generator for the swarm game."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from random import Random
from typing import List

from .config import ScenarioBundle
from .entities import ADUnit, Drone, GridPosition, Interceptor, TargetCluster


@dataclass
class SwarmScenario:
    scenario_id: str
    attackers: List[Drone]
    interceptors: List[Interceptor]
    ad_units: List[ADUnit]
    targets: List[TargetCluster]
    bundle: ScenarioBundle


class ScenarioGenerator:
    def __init__(self, bundle: ScenarioBundle | None = None):
        self.bundle = bundle or ScenarioBundle()
        self._rng = Random(self.bundle.scenario.seed)

    def generate(self) -> SwarmScenario:
        scenario_id = str(uuid.uuid4())
        targets = self._generate_targets()
        attackers = self._generate_attackers(targets)
        interceptors = self._generate_interceptors()
        ad_units = self._generate_ad_units()
        return SwarmScenario(
            scenario_id=scenario_id,
            attackers=attackers,
            interceptors=interceptors,
            ad_units=ad_units,
            targets=targets,
            bundle=self.bundle,
        )

    # ------------------------------------------------------------------
    # Helpers
    def _generate_targets(self) -> List[TargetCluster]:
        grid = self.bundle.grid
        targets: List[TargetCluster] = []
        bottom_start = grid.size // 2
        for idx, value in enumerate(self.bundle.scenario.target_values):
            x = self._rng.randint(0, grid.size - 1)
            y = self._rng.randint(bottom_start, grid.size - 1)
            position = GridPosition(x, y)
            target = TargetCluster(
                cluster_id=f"T{idx+1}",
                position=position,
                value=value,
                tot_offset=self.bundle.scenario.tot_offsets[idx],
                radius=1 + idx % 2,
            )
            targets.append(target)
        return targets

    def _generate_attackers(self, targets: List[TargetCluster]) -> List[Drone]:
        grid = self.bundle.grid
        num_attackers = self.bundle.scenario.num_attackers
        attackers: List[Drone] = []
        spacing = grid.size / (num_attackers + 1)
        for idx in range(num_attackers):
            x = int((idx + 1) * spacing)
            attacker_position = GridPosition(x, 0)
            target = self._rng.choice(targets)
            next_position = attacker_position.step_towards(target.position)
            attackers.append(
                Drone(
                    drone_id=f"A{idx+1}",
                    position=attacker_position,
                    next_position=next_position,
                    target_cluster=target.cluster_id,
                    tot_offset=target.tot_offset,
                    payload_value=target.value,
                )
            )
        return attackers

    def _generate_interceptors(self) -> List[Interceptor]:
        grid = self.bundle.grid
        interceptors: List[Interceptor] = []
        for idx in range(self.bundle.scenario.num_interceptors):
            x = self._rng.randint(0, grid.size - 1)
            y = self._rng.randint(grid.size // 4, (grid.size // 4) * 3)
            interceptors.append(
                Interceptor(
                    interceptor_id=f"I{idx+1}",
                    position=GridPosition(x, y),
                )
            )
        return interceptors

    def _generate_ad_units(self) -> List[ADUnit]:
        grid = self.bundle.grid
        stride = self.bundle.grid.ad_stride
        candidates = [
            GridPosition(x, y)
            for x in range(stride // 2, grid.size, stride)
            for y in range(stride // 2, grid.size, stride)
        ]
        self._rng.shuffle(candidates)
        ad_units: List[ADUnit] = []
        for idx in range(self.bundle.scenario.num_ad_units):
            position = candidates[idx % len(candidates)]
            ad_units.append(
                ADUnit(
                    ad_id=f"AD{idx+1}",
                    position=position,
                    coverage=2 + idx,
                )
            )
        return ad_units
