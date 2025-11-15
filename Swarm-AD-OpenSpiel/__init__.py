"""Swarm Air-Defense OpenSpiel utilities."""

try:  # Prefer relative import when the package context is available.
    from .swarm_defense_game import (
        ADUnit,
        SwarmDefenseGame,
        SwarmDefenseState,
        TargetCluster,
        DronePlan,
        Phase,
        TOT_CHOICES,
    )
except ImportError:  # Fallback for pytest collecting the file as a loose module.
    from swarm_defense_game import (  # type: ignore
        ADUnit,
        SwarmDefenseGame,
        SwarmDefenseState,
        TargetCluster,
        DronePlan,
        Phase,
        TOT_CHOICES,
    )

__all__ = [
    "SwarmDefenseGame",
    "SwarmDefenseState",
    "TargetCluster",
    "ADUnit",
    "DronePlan",
    "Phase",
    "TOT_CHOICES",
]
