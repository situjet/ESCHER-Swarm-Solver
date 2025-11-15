"""Configuration dataclasses for the swarm visualizer."""

from __future__ import annotations

from typing import Dict, List, Tuple

from pydantic import BaseModel, Field, validator


class GridConfig(BaseModel):
    size: int = Field(16, description="Length of one grid side (square grid).")
    ad_stride: int = Field(8, description="Stride used to place AD units on lattice points.")
    cell_size_deg: float = Field(0.00012, description="Lat/Lon delta per cell for display.")

    @validator("size")
    def validate_size(cls, value: int) -> int:  # noqa: D401
        """Ensure the grid side length is positive and even (to split halves)."""
        if value <= 0 or value % 2 != 0:
            raise ValueError("grid size must be a positive even integer")
        return value


class ScenarioConfig(BaseModel):
    num_attackers: int = 6
    num_interceptors: int = 3
    num_ad_units: int = 2
    target_values: Tuple[int, int, int] = (10, 20, 40)
    tot_offsets: Tuple[int, int, int] = (0, 2, 4)
    ad_kill_probability: float = 0.5
    seed: int | None = None

    @validator("ad_kill_probability")
    def validate_probability(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("AD kill probability must be between 0 and 1")
        return value


class GeoConfig(BaseModel):
    origin_lat: float = 34.000000
    origin_lon: float = -117.000000
    altitude_ft: int = 100


class PyTakRuntimeConfig(BaseModel):
    cot_endpoint: str = Field("udp://127.0.0.1:6969", description="CoT endpoint for WinTAK")
    cot_callsign_prefix: str = Field("SWARM", description="Prefix for UID/callsigns.")
    dry_run: bool = Field(False, description="Print CoT without transmitting.")
    export_file: str | None = Field(None, description="Optional path to write CoT stream.")
    step_delay: float = Field(1.0, description="Seconds between CoT snapshots.")

    def to_pytak_config(self) -> Dict[str, str]:
        return {
            "COT_URL": self.cot_endpoint,
            "PYTAK_TLS": "0",
        }


class ScenarioBundle(BaseModel):
    grid: GridConfig = Field(default_factory=GridConfig)
    scenario: ScenarioConfig = Field(default_factory=ScenarioConfig)
    geo: GeoConfig = Field(default_factory=GeoConfig)

    def target_tot_table(self) -> Dict[int, int]:
        return {idx: offset for idx, offset in enumerate(self.scenario.tot_offsets)}


DEFAULT_BUNDLE = ScenarioBundle()
DEFAULT_PYTAK_RUNTIME = PyTakRuntimeConfig()
