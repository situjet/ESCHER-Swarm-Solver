"""Cursor-on-Target helpers for WinTAK visualization."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, List, Sequence

from .config import GeoConfig, GridConfig, PyTakRuntimeConfig
from .entities import DroneStatus
from .history import GameStateSnapshot

COT_XML_TEMPLATE = (
    '<event version="2.0" uid="{uid}" type="{cot_type}" time="{time}" start="{time}" stale="{stale}" how="m-g">'
    "<point lat=\"{lat:.6f}\" lon=\"{lon:.6f}\" hae=\"{hae}\" ce=\"{ce}\" le=\"{le}\"/>"
    "<detail><contact callsign=\"{callsign}\"/>"
    "<remarks>{remarks}</remarks>"
    "</detail></event>"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def grid_to_latlon(x: int, y: int, grid: GridConfig, geo: GeoConfig) -> tuple[float, float]:
    lat = geo.origin_lat - (y * grid.cell_size_deg)
    lon = geo.origin_lon + (x * grid.cell_size_deg)
    return lat, lon


def format_cot_event(
    uid: str,
    callsign: str,
    cot_type: str,
    lat: float,
    lon: float,
    *,
    hae: int = 100,
    ce: int = 20,
    le: int = 20,
    remarks: str = "",
) -> str:
    timestamp = _now_iso()
    stale_time = _now_iso()
    return COT_XML_TEMPLATE.format(
        uid=uid,
        cot_type=cot_type,
        time=timestamp,
        stale=stale_time,
        lat=lat,
        lon=lon,
        hae=hae,
        ce=ce,
        le=le,
        callsign=callsign,
        remarks=remarks,
    )


def snapshot_to_cot(
    snapshot: GameStateSnapshot,
    *,
    grid: GridConfig,
    geo: GeoConfig,
    runtime: PyTakRuntimeConfig,
) -> List[str]:
    events: List[str] = []
    prefix = runtime.cot_callsign_prefix
    for drone in snapshot.drones:
        lat, lon = grid_to_latlon(drone.x, drone.y, grid, geo)
        cot_type = "a-f-A-M-F"
        remarks = f"Drone {drone.drone_id} -> {drone.target_cluster}, TOT+{drone.tot_offset}s, status={drone.status.value}"
        if drone.status is DroneStatus.COMPLETED:
            cot_type = "a-f-A-M-F-C"
        elif drone.status is DroneStatus.DESTROYED:
            cot_type = "a-f-A-M-F-K"
            if drone.kill_source:
                remarks += f" | killed by {drone.kill_source}"
        events.append(
            format_cot_event(
                uid=f"{prefix}-{drone.drone_id}",
                callsign=f"{prefix}-{drone.drone_id}",
                cot_type=cot_type,
                lat=lat,
                lon=lon,
                remarks=remarks,
            )
        )
        if drone.kill_source and drone.kill_x is not None and drone.kill_y is not None:
            kill_lat, kill_lon = grid_to_latlon(drone.kill_x, drone.kill_y, grid, geo)
            kill_tick = f" @t={drone.kill_tick}" if drone.kill_tick is not None else ""
            events.append(
                format_cot_event(
                    uid=f"{prefix}-{drone.drone_id}-kill",
                    callsign=f"{prefix}-{drone.drone_id}-K",
                    cot_type="b-m-p-s-k",
                    lat=kill_lat,
                    lon=kill_lon,
                    remarks=f"Kill site for {drone.drone_id}{kill_tick} via {drone.kill_source}",
                    hae=0,
                    ce=10,
                    le=10,
                )
            )

    for interceptor in snapshot.interceptors:
        lat, lon = grid_to_latlon(interceptor.x, interceptor.y, grid, geo)
        remarks = f"Interceptor {interceptor.interceptor_id}, tracking={interceptor.assigned_drone or 'NONE'}"
        events.append(
            format_cot_event(
                uid=f"{prefix}-{interceptor.interceptor_id}",
                callsign=f"{prefix}-{interceptor.interceptor_id}",
                cot_type="a-f-A-M-H",
                lat=lat,
                lon=lon,
                remarks=remarks,
                hae=150,
            )
        )

    for ad in snapshot.ad_units:
        lat, lon = grid_to_latlon(ad.x, ad.y, grid, geo)
        remarks = f"AD {ad.ad_id}, coverage={ad.coverage}, engaged={','.join(ad.engaged_drones) or 'none'}"
        events.append(
            format_cot_event(
                uid=f"{prefix}-{ad.ad_id}",
                callsign=f"{prefix}-{ad.ad_id}",
                cot_type="a-f-A-M-D",
                lat=lat,
                lon=lon,
                remarks=remarks,
                hae=50,
            )
        )

    for target in snapshot.targets:
        lat, lon = grid_to_latlon(target.x, target.y, grid, geo)
        status = "destroyed" if target.is_destroyed else "active"
        remarks = f"Target {target.cluster_id} ({target.value}) TOT+{target.tot_offset}s {status}"
        cot_type = "b-m-p-s-m"
        events.append(
            format_cot_event(
                uid=f"{prefix}-{target.cluster_id}",
                callsign=f"{prefix}-{target.cluster_id}",
                cot_type=cot_type,
                lat=lat,
                lon=lon,
                remarks=remarks,
                hae=0,
            )
        )
    return events


def iter_history_as_cot(
    history: Sequence[GameStateSnapshot],
    grid: GridConfig,
    geo: GeoConfig,
    runtime: PyTakRuntimeConfig,
) -> Iterable[List[str]]:
    for snapshot in history:
        yield snapshot_to_cot(snapshot, grid=grid, geo=geo, runtime=runtime)
