"""Cursor-on-Target formatting utilities for WinTAK visualization."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Tuple

# Pittsburgh International Airport coordinates
PITTSBURGH_AIRPORT_LAT = 40.4915
PITTSBURGH_AIRPORT_LON = -80.2329

# Arena dimensions (matching Large game: 32x32)
ARENA_WIDTH = 32.0
ARENA_HEIGHT = 32.0

# Degrees per unit for geographic mapping
# 6x scale fan-out so icons are easier to distinguish on the map
DEGREES_PER_UNIT = 0.0006

COT_XML_TEMPLATE = (
    '<event version="2.0" uid="{uid}" type="{cot_type}" time="{time}" start="{start}" stale="{stale}" how="m-g">'
    "<point lat=\"{lat:.6f}\" lon=\"{lon:.6f}\" hae=\"{hae}\" ce=\"{ce}\" le=\"{le}\"/>"
    "<detail><contact callsign=\"{callsign}\"/>"
    "{extra_detail}"
    "<remarks>{remarks}</remarks>"
    "</detail></event>"
)


def _cot_timestamp(dt: datetime) -> str:
    """Return a TAK-friendly ISO-8601 timestamp with millisecond precision."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def arena_to_latlon(y: float, x: float) -> Tuple[float, float]:
    """
    Convert arena coordinates to lat/lon centered on Pittsburgh Airport.
    
    Args:
        y: Row position in arena (0 = top, ARENA_HEIGHT = bottom)
        x: Column position in arena (0 = left, ARENA_WIDTH = right)
    
    Returns:
        (latitude, longitude) tuple
    """
    # Center the arena on the airport
    # Y increases downward in arena, but latitude increases northward
    lat = PITTSBURGH_AIRPORT_LAT + ((ARENA_HEIGHT / 2.0) - y) * DEGREES_PER_UNIT
    lon = PITTSBURGH_AIRPORT_LON + (x - (ARENA_WIDTH / 2.0)) * DEGREES_PER_UNIT
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
    stale_seconds: float = 120.0,
    extra_detail: str = "",
) -> str:
    """
    Format a single CoT event.
    
    Args:
        uid: Unique identifier for the event
        callsign: Callsign displayed in WinTAK
        cot_type: 2525-B symbol type code
        lat: Latitude
        lon: Longitude
        hae: Height above ellipsoid (meters)
        ce: Circular error (meters)
        le: Linear error (meters)
        remarks: Additional remarks text
    
    Returns:
        CoT XML string
    """
    now = datetime.now(timezone.utc)
    start_time = now
    stale_time = now + timedelta(seconds=max(stale_seconds, 0.1))
    return COT_XML_TEMPLATE.format(
        uid=uid,
        cot_type=cot_type,
        time=_cot_timestamp(now),
        start=_cot_timestamp(start_time),
        stale=_cot_timestamp(stale_time),
        lat=lat,
        lon=lon,
        hae=hae,
        ce=ce,
        le=le,
        callsign=callsign,
        remarks=remarks,
        extra_detail=extra_detail,
    )


def wrap_xml(body: str) -> bytes:
    """Wrap CoT XML body with XML declaration."""
    xml_declaration = '<?xml version="1.0" encoding="utf-8" standalone="yes"?>\n'
    return (xml_declaration + body).encode("utf-8")


# CoT type mappings for different entity types
COT_TYPES = {
    "drone_active": "a-h-A-M-F",  # REDFOR swarm aircraft icon
    "interceptor": "a-f-A-M-H",  # BLUFOR interceptor
    "ad_unit": "a-f-G-U-C",  # BLUFOR ground-based air defense
    "target": "a-n-A-M-F",  # GREEN/INDFOR target markers
    "ad_engagement": "b-m-p-s-p-l",  # Engagement line indicator
    "ad_fov": "b-m-p-s-p-r",  # Sensor/radar range fan
    "delete": "t-x-d-d",  # TAK delete directive
}


def format_delete_event(uid: str, remarks: str = "Removed") -> str:
    """Create a CoT delete directive for a specific UID."""

    return format_cot_event(
        uid=uid,
        callsign=uid,
        cot_type=COT_TYPES["delete"],
        lat=0.0,
        lon=0.0,
        hae=0,
        ce=999999,
        le=999999,
        remarks=remarks,
        stale_seconds=1.0,
    )


# Color coding for remarks (for context, not directly used in CoT)
TOT_COLORS = {
    0.0: "RED",
    1.5: "ORANGE", 
    3.0: "YELLOW",
    4.5: "PURPLE",
}
