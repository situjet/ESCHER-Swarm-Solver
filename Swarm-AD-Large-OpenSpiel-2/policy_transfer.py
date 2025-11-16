"""Legacy blueprint transfer helpers have been removed.

The second large-scale Swarm Air-Defense environment now operates independently,
without relying on policy lifting from the abstract grid game.  This module is
kept as a compatibility shim so that older scripts importing
``Swarm-AD-Large-OpenSpiel-2.policy_transfer`` fail fast with a clear message
rather than crashing deep inside a rollout.
"""

from __future__ import annotations


def __getattr__(name: str):  # pragma: no cover - defensive compatibility hook
    raise ImportError(
        "policy_transfer has been removed from Swarm-AD-Large-OpenSpiel-2. "
        "Interact with the large game directly via swarm_defense_large_game."
    )


__all__ = []
