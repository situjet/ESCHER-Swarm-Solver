"""Unit tests for the swarm scenario generator and simulator."""

from swarm_visualizer.config import ScenarioBundle, ScenarioConfig
from swarm_visualizer.entities import DroneStatus
from swarm_visualizer.scenario import ScenarioGenerator
from swarm_visualizer.simulator import SwarmGameSimulator


def _bundle(seed: int = 42) -> ScenarioBundle:
    return ScenarioBundle(scenario=ScenarioConfig(seed=seed))


def test_targets_spawn_in_bottom_half():
    generator = ScenarioGenerator(_bundle())
    scenario = generator.generate()
    grid_mid = scenario.bundle.grid.size // 2
    assert len(scenario.targets) == 3
    assert all(target.position.y >= grid_mid for target in scenario.targets)


def test_ad_units_follow_stride():
    bundle = _bundle()
    generator = ScenarioGenerator(bundle)
    scenario = generator.generate()
    stride = bundle.grid.ad_stride
    allowed_coords = {((stride // 2) + stride * a, (stride // 2) + stride * b) for a in range(2) for b in range(2)}
    assert all((ad.position.x, ad.position.y) in allowed_coords for ad in scenario.ad_units)


def test_simulator_resolves_all_attackers():
    bundle = _bundle(13)
    generator = ScenarioGenerator(bundle)
    scenario = generator.generate()
    simulator = SwarmGameSimulator(scenario, bundle=bundle)
    history = simulator.run(max_ticks=50)
    assert history.snapshots[-1].tick <= 50
    assert all(drone.status is not DroneStatus.ACTIVE for drone in history.snapshots[-1].drones)
