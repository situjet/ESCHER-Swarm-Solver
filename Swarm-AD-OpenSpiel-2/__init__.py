"""2Swarm2 OpenSpiel utilities."""

try:
    from .Swarm_AD_OpenSpiel_2.two_swarm2_game import (
        ADUnit,
        DronePlan,
        Phase,
        TargetCluster,
        TOT_CHOICES,
        TwoSwarm2Game,
        TwoSwarm2State,
    )
except ImportError:  # pragma: no cover - fallback for module execution
    from Swarm_AD_OpenSpiel_2.two_swarm2_game import (  # type: ignore
        ADUnit,
        DronePlan,
        Phase,
        TargetCluster,
        TOT_CHOICES,
        TwoSwarm2Game,
        TwoSwarm2State,
    )

__all__ = [
    "TwoSwarm2Game",
    "TwoSwarm2State",
    "TargetCluster",
    "ADUnit",
    "DronePlan",
    "Phase",
    "TOT_CHOICES",
]
