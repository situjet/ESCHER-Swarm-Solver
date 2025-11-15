"""Send one or more Cursor-on-Target events directly to a TAK endpoint."""
from __future__ import annotations

import argparse
import socket
import ssl
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

from swarm_visualizer.cot import format_cot_event

SUPPORTED_SCHEMES = {"udp", "udp+wo", "tcp", "tls"}
XML_DECLARATION = '<?xml version="1.0" encoding="utf-8" standalone="yes"?>\n'


@dataclass(frozen=True)
class BuiltPayload:
    uid: str
    callsign: str
    cot_type: str
    lat: float
    lon: float
    payload: bytes


@dataclass(frozen=True)
class IconSpec:
    uid: str
    callsign: str
    cot_type: str
    remarks: str
    hae: int = 100
    ce: int = 20
    le: int = 20
    lat: float | None = None
    lon: float | None = None
    lat_offset: float = 0.0
    lon_offset: float = 0.0


SAMPLE_ICON_PACK: List[IconSpec] = [
    IconSpec(
        uid="ICON-MONUMENT-RED",
        callsign="RED-UAS",
        cot_type="a-h-A-M-U",
        remarks="Hostile quadcopter loitering over Washington Monument",
        hae=400,
    ),
    IconSpec(
        uid="ICON-WHITEHOUSE-RED",
        callsign="RED-JET",
        cot_type="a-h-A-M-F",
        remarks="Hostile fixed-wing inbound from northwest",
        hae=800,
        ce=30,
        le=30,
        lat_offset=0.0082,
        lon_offset=-0.0012,
    ),
    IconSpec(
        uid="ICON-CAPITOL-TGT",
        callsign="TGT-ALPHA",
        cot_type="b-m-p-s-m",
        remarks="High-value target cluster at U.S. Capitol",
        hae=0,
        ce=15,
        le=15,
        lat_offset=0.0004,
        lon_offset=0.0262,
    ),
    IconSpec(
        uid="ICON-LINCOLN-KILL",
        callsign="KILL-SITE",
        cot_type="b-m-p-s-k",
        remarks="Recent kill site near Lincoln Memorial",
        hae=0,
        ce=10,
        le=10,
        lat_offset=-0.0002,
        lon_offset=-0.0149,
    ),
    IconSpec(
        uid="ICON-PENTAGON-BLU",
        callsign="BLUE-INT",
        cot_type="a-f-A-M-H",
        remarks="Friendly interceptor orbiting Pentagon",
        hae=500,
        lat_offset=-0.0176,
        lon_offset=-0.0210,
    ),
    IconSpec(
        uid="ICON-BOLLING-AD",
        callsign="BLUE-AD",
        cot_type="a-f-A-M-D",
        remarks="Air defense battery at JBAB",
        hae=100,
        ce=25,
        le=25,
        lat_offset=-0.0391,
        lon_offset=0.0464,
    ),
]


def _send_udp(host: str, port: int, payload: bytes) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.sendto(payload, (host, port))


def _send_tcp(host: str, port: int, payload: bytes, use_tls: bool) -> None:
    with socket.create_connection((host, port), timeout=5) as raw_sock:
        if use_tls:
            context = ssl.create_default_context()
            with context.wrap_socket(raw_sock, server_hostname=host) as tls_sock:
                tls_sock.sendall(payload)
        else:
            raw_sock.sendall(payload)


SENDERS: dict[str, Callable[[str, int, bytes], None]] = {
    "udp": lambda host, port, payload: _send_udp(host, port, payload),
    "udp+wo": lambda host, port, payload: _send_udp(host, port, payload),
    "tcp": lambda host, port, payload: _send_tcp(host, port, payload, use_tls=False),
    "tls": lambda host, port, payload: _send_tcp(host, port, payload, use_tls=True),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Push synthetic CoT event to a TAK endpoint")
    parser.add_argument("--cot-endpoint", required=True, help="Endpoint URL, e.g. udp://127.0.0.1:6969")
    parser.add_argument("--mode", choices=("single", "sample-pack"), default="single", help="Send a single icon or the curated sample pack")
    parser.add_argument("--uid", default="TEST-DRONE", help="Unique ID for the CoT event (single mode)")
    parser.add_argument("--callsign", default="TEST-DRONE", help="Callsign placed in <contact> (single mode)")
    parser.add_argument("--cot-type", default="a-f-A-M-F", help="2525-B symbol type code (single mode)")
    parser.add_argument("--lat", type=float, default=38.8895, help="Latitude anchor (or base for sample pack)")
    parser.add_argument("--lon", type=float, default=-77.0353, help="Longitude anchor (or base for sample pack)")
    parser.add_argument("--hae", type=int, default=100, help="Height above ellipsoid")
    parser.add_argument("--ce", type=int, default=20, help="Circular error")
    parser.add_argument("--le", type=int, default=20, help="Linear error")
    parser.add_argument("--remarks", default="Swarm visualizer test ping", help="Remarks text")
    parser.add_argument("--count", type=int, default=1, help="How many batches to send")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between batches when count>1")
    return parser.parse_args(argv)


def _wrap_xml(body: str) -> bytes:
    return (XML_DECLARATION + body).encode("utf-8")


def build_single_payload(args: argparse.Namespace) -> BuiltPayload:
    body = format_cot_event(
        uid=args.uid,
        callsign=args.callsign,
        cot_type=args.cot_type,
        lat=args.lat,
        lon=args.lon,
        hae=args.hae,
        ce=args.ce,
        le=args.le,
        remarks=args.remarks,
    )
    return BuiltPayload(
        uid=args.uid,
        callsign=args.callsign,
        cot_type=args.cot_type,
        lat=args.lat,
        lon=args.lon,
        payload=_wrap_xml(body),
    )


def build_sample_payloads(args: argparse.Namespace) -> List[BuiltPayload]:
    payloads: List[BuiltPayload] = []
    for spec in SAMPLE_ICON_PACK:
        lat = spec.lat if spec.lat is not None else args.lat + spec.lat_offset
        lon = spec.lon if spec.lon is not None else args.lon + spec.lon_offset
        body = format_cot_event(
            uid=spec.uid,
            callsign=spec.callsign,
            cot_type=spec.cot_type,
            lat=lat,
            lon=lon,
            hae=spec.hae,
            ce=spec.ce,
            le=spec.le,
            remarks=spec.remarks,
        )
        payloads.append(
            BuiltPayload(
                uid=spec.uid,
                callsign=spec.callsign,
                cot_type=spec.cot_type,
                lat=lat,
                lon=lon,
                payload=_wrap_xml(body),
            )
        )
    return payloads


def build_payloads(args: argparse.Namespace) -> List[BuiltPayload]:
    if args.mode == "sample-pack":
        return build_sample_payloads(args)
    return [build_single_payload(args)]


def send_events(args: argparse.Namespace) -> None:
    parsed = urlparse(args.cot_endpoint)
    scheme = (parsed.scheme or "").lower()
    if scheme not in SUPPORTED_SCHEMES:
        raise SystemExit(f"Unsupported scheme '{scheme}'. Use one of: {', '.join(sorted(SUPPORTED_SCHEMES))}.")
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if scheme == "tls" else 6969)
    sender = SENDERS[scheme]
    payloads = build_payloads(args)
    total_icons = len(payloads)
    if not payloads:
        return
    for batch_idx in range(args.count):
        for icon_idx, built in enumerate(payloads, start=1):
            sender(host, port, built.payload)
            print(
                f"[batch {batch_idx+1}/{args.count} icon {icon_idx}/{total_icons}] "
                f"Sent {len(built.payload)} bytes for {built.uid} ({built.cot_type}) "
                f"at lat={built.lat:.6f}, lon={built.lon:.6f} to {scheme}://{host}:{port}"
            )
        if batch_idx + 1 < args.count:
            time.sleep(args.interval)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        send_events(args)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to send CoT payload: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
