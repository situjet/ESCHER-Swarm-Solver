"""Stream visualize_policy snapshots to WinTAK using the proven CoT format."""
from __future__ import annotations

import argparse
import socket
import ssl
import sys
import uuid
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SWARM_OSP_ROOT = PROJECT_ROOT / "Swarm-AD-OpenSpiel"
ESCHER_TORCH_ROOT = PROJECT_ROOT / "ESCHER-Torch"
WINTAK_ROOT = PROJECT_ROOT / "WinTAK-CoT-Generator"

for candidate in (SWARM_OSP_ROOT, ESCHER_TORCH_ROOT, WINTAK_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cot_helpers import (  # type: ignore  # noqa: E402
    COT_TYPES,
    TOT_COLORS,
    arena_to_latlon,
    format_cot_event,
    wrap_xml,
)
from swarm_defense_game import AD_COVERAGE_RADIUS  # type: ignore  # noqa: E402
from visualize_policy import play_episode_with_policy  # type: ignore  # noqa: E402

SUPPORTED_SCHEMES = {"udp", "udp+wo", "tcp", "tls"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play visualize_policy and push its snapshot to WinTAK using the trusted CoT helpers."
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        help="Path to a precomputed wintak_snapshot.json; skips policy rollout",
    )
    parser.add_argument("--policy", type=str, help="Single checkpoint applied to both players")
    parser.add_argument("--policy-p0", type=str, help="Checkpoint for attacker (player 0)")
    parser.add_argument("--policy-p1", type=str, help="Checkpoint for defender (player 1)")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed")
    parser.add_argument("--sampling", action="store_true", help="Sample from policy distribution")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="Inference device")
    parser.add_argument("--scenario-id", type=str, default=None, help="Scenario label for UID metadata")

    parser.add_argument(
        "--cot-endpoint",
        default="tcp://127.0.0.1:6969",
        help="CoT endpoint URI (e.g., tcp://host:port)",
    )
    parser.add_argument("--callsign-prefix", default="SWARM", help="Prefix for UID/callsigns")
    parser.add_argument("--dry-run", action="store_true", help="Print payloads instead of sending")
    parser.add_argument("--export-file", type=str, default=None, help="Optional path to save CoT XML stream")
    return parser.parse_args()


def _resolve_policy_paths(args: argparse.Namespace) -> Tuple[Optional[Path], Optional[Path]]:
    if args.snapshot:
        raise SystemExit("--snapshot provided; policy checkpoints should be omitted")

    if args.policy:
        path = Path(args.policy).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"Policy file not found: {path}")
        return path, path

    p0 = Path(args.policy_p0).expanduser().resolve() if args.policy_p0 else None
    p1 = Path(args.policy_p1).expanduser().resolve() if args.policy_p1 else None
    if p0 is None and p1 is None:
        raise SystemExit("Provide --policy, --policy-p0, or --policy-p1")
    for candidate in (p0, p1):
        if candidate is not None and not candidate.exists():
            raise SystemExit(f"Policy file not found: {candidate}")
    return p0, p1


def _matchup_label(p0: Optional[Path], p1: Optional[Path]) -> str:
    if p0 and p1:
        return "policy-v-policy"
    if p0 and not p1:
        return "policy-v-naive"
    if not p0 and p1:
        return "naive-v-policy"
    return "naive-v-naive"


def _load_snapshot(snapshot_arg: str) -> Tuple[Dict[str, Any], Path]:
    snapshot_path = Path(snapshot_arg).expanduser().resolve()
    if not snapshot_path.exists():
        raise SystemExit(f"Snapshot file not found: {snapshot_path}")

    with snapshot_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit("Snapshot JSON must be an object")
    return payload, snapshot_path


def _extract_row_col(point: Sequence[float] | None) -> Optional[Tuple[float, float]]:
    if not point:
        return None
    row, col = point
    return float(row), float(col)


def _drone_display_position(drone_info: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    destroyed_by = drone_info.get("destroyed_by")
    if destroyed_by == "interceptor" and drone_info.get("interceptor_hit"):
        return _extract_row_col(drone_info.get("interceptor_hit"))
    intercepts = drone_info.get("intercepts") or []
    if intercepts:
        _, point, _ = intercepts[-1]
        return _extract_row_col(point)
    if drone_info.get("destination"):
        return _extract_row_col(drone_info.get("destination"))
    if drone_info.get("entry"):
        return _extract_row_col(drone_info.get("entry"))
    return None


def _target_label(index: int, num_targets: int) -> str:
    return f"T{index}" if index < num_targets else f"AD{index - num_targets}"


def _build_drone_events(snapshot: Dict[str, Any], prefix: str) -> Tuple[List[str], int]:
    events: List[str] = []
    targets = snapshot.get("targets", ())
    for idx, drone_info in enumerate(snapshot.get("drones", ())):
        position = _drone_display_position(drone_info)
        if position is None:
            continue
        row, col = position
        lat, lon = arena_to_latlon(row, col)
        tot_value = float(drone_info.get("tot") or 0.0)
        tot_color = TOT_COLORS.get(tot_value, "UNKNOWN")
        target_idx = int(drone_info.get("target_idx", 0))
        target_tag = _target_label(target_idx, len(targets))
        target_value = drone_info.get("target_value")
        destroyed_by = drone_info.get("destroyed_by")

        status = "DESTROYED" if destroyed_by else "ACTIVE"
        if not destroyed_by and drone_info.get("strike_success") is True:
            status = "SUCCESS"
        elif not destroyed_by and drone_info.get("strike_success") is False:
            status = "FAILED"

        remarks = (
            f"Drone D{idx} -> {target_tag}, TOT+{tot_value:.1f}s ({tot_color}), "
            f"status={status}, target_value={target_value}"
        )
        if destroyed_by:
            remarks += f" | threat={destroyed_by}"

        uid = f"{prefix}-DRONE-{idx}"
        events.append(
            format_cot_event(
                uid=uid,
                callsign=f"{prefix}-D{idx}",
                cot_type=COT_TYPES["drone_active"],
                lat=lat,
                lon=lon,
                hae=100,
                ce=15,
                le=15,
                remarks=remarks,
            )
        )
    return events, len(events)


def _build_ad_events(snapshot: Dict[str, Any], prefix: str) -> Tuple[List[str], int]:
    events: List[str] = []
    for idx, ad_info in enumerate(snapshot.get("ad_units", ())):
        if not ad_info.get("alive", True):
            continue
        position = _extract_row_col(ad_info.get("position"))
        if position is None:
            continue
        row, col = position
        lat, lon = arena_to_latlon(row, col)
        engaged_ids = sorted({entry[0] for entry in ad_info.get("intercept_log", ())})
        engaged = ",".join(f"D{drone_id}" for drone_id in engaged_ids) or "none"
        coverage = float(ad_info.get("coverage", AD_COVERAGE_RADIUS))
        remarks = f"AD Unit {idx}: coverage≈{coverage:.1f}, engaged={engaged}"

        uid = f"{prefix}-AD-{idx}"
        events.append(
            format_cot_event(
                uid=uid,
                callsign=f"{prefix}-AD{idx}",
                cot_type=COT_TYPES["ad_unit"],
                lat=lat,
                lon=lon,
                hae=50,
                ce=25,
                le=25,
                remarks=remarks,
            )
        )
    return events, len(events)


def _build_target_events(snapshot: Dict[str, Any], prefix: str) -> Tuple[List[str], int]:
    events: List[str] = []
    destroyed_flags = snapshot.get("target_destroyed", ())
    targets = snapshot.get("targets", ())
    for idx, target in enumerate(targets):
        row = getattr(target, "row", None)
        col = getattr(target, "col", None)
        if row is None or col is None:
            continue
        lat, lon = arena_to_latlon(float(row), float(col))
        destroyed = bool(destroyed_flags[idx]) if idx < len(destroyed_flags) else False
        status = "DESTROYED" if destroyed else "ACTIVE"
        value = getattr(target, "value", 0)
        remarks = f"Target T{idx}: value={value:.0f}, status={status}"

        uid = f"{prefix}-TGT-{idx}"
        events.append(
            format_cot_event(
                uid=uid,
                callsign=f"{prefix}-T{idx}",
                cot_type=COT_TYPES["target"],
                lat=lat,
                lon=lon,
                hae=0,
                ce=20,
                le=20,
                remarks=remarks,
            )
        )
    return events, len(events)


def build_cot_events(snapshot: Dict[str, Any], prefix: str) -> Tuple[List[str], Dict[str, int]]:
    events: List[str] = []
    counts: Dict[str, int] = {}

    drone_events, drone_count = _build_drone_events(snapshot, prefix)
    ad_events, ad_count = _build_ad_events(snapshot, prefix)
    target_events, target_count = _build_target_events(snapshot, prefix)

    events.extend(drone_events)
    events.extend(ad_events)
    events.extend(target_events)

    counts.update({"drones": drone_count, "ad_units": ad_count, "targets": target_count})
    return events, counts


def _send_udp(host: str, port: int, payload: bytes) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.sendto(payload, (host, port))


def _send_tcp(host: str, port: int, payload: bytes, *, use_tls: bool = False) -> None:
    with socket.create_connection((host, port), timeout=5) as raw_sock:
        if use_tls:
            context = ssl.create_default_context()
            with context.wrap_socket(raw_sock, server_hostname=host) as tls_sock:
                tls_sock.sendall(payload)
        else:
            raw_sock.sendall(payload)


def _resolve_sender(scheme: str):
    if scheme == "udp" or scheme == "udp+wo":
        return lambda host, port, payload: _send_udp(host, port, payload)
    if scheme == "tcp":
        return lambda host, port, payload: _send_tcp(host, port, payload, use_tls=False)
    if scheme == "tls":
        return lambda host, port, payload: _send_tcp(host, port, payload, use_tls=True)
    raise ValueError(f"Unsupported scheme: {scheme}")


def send_cot_events(events: List[str], endpoint: str, *, dry_run: bool, export_file: Optional[str]) -> None:
    parsed = urlparse(endpoint)
    scheme = (parsed.scheme or "").lower()
    if scheme not in SUPPORTED_SCHEMES:
        raise SystemExit(
            f"Unsupported scheme '{scheme}'. Use one of: {', '.join(sorted(SUPPORTED_SCHEMES))}."
        )
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 6969
    sender = _resolve_sender(scheme)

    xml_payloads: List[str] = []
    for idx, event in enumerate(events, start=1):
        payload = wrap_xml(event)
        if dry_run:
            print(f"[dry-run {idx}/{len(events)}] {len(payload)} bytes -> {host}:{port}\n{event}\n")
        else:
            sender(host, port, payload)
        xml_payloads.append(payload.decode("utf-8"))

    if export_file:
        Path(export_file).write_text("\n".join(xml_payloads), encoding="utf-8")
        print(f"Saved CoT stream to {export_file}")

    print(f"Dispatched {len(events)} CoT event(s) to {scheme}://{host}:{port} (dry_run={dry_run})")


def main() -> None:
    args = parse_args()
    snapshot: Dict[str, Any]
    seed: Optional[int]
    return_values: Optional[Sequence[float]] = None

    if args.snapshot:
        snapshot, snapshot_path = _load_snapshot(args.snapshot)
        policy_p0 = policy_p1 = None
        matchup = args.scenario_id or snapshot_path.stem
        seed = snapshot.get("seed")
    else:
        policy_p0, policy_p1 = _resolve_policy_paths(args)
        matchup = args.scenario_id or _matchup_label(policy_p0, policy_p1)
        state, seed = play_episode_with_policy(
            policy_path_p0=policy_p0,
            policy_path_p1=policy_p1,
            seed=args.seed,
            use_sampling=args.sampling,
            device=args.device,
        )
        snapshot = state.snapshot()
        return_values = state.returns()

    if return_values is None:
        raw_returns = snapshot.get("returns")
        if isinstance(raw_returns, Sequence):
            return_values = raw_returns

    prefix = args.callsign_prefix.upper().strip() or "SWARM"
    events, counts = build_cot_events(snapshot, prefix)
    send_cot_events(events, args.cot_endpoint, dry_run=args.dry_run, export_file=args.export_file)

    print("-" * 60)
    print(f"Scenario: {matchup}")
    print(f"Seed: {seed}")
    if return_values:
        print(f"Attacker damage: {return_values[0]:.2f}")
        print(f"Defender reward: {return_values[1]:.2f}")
    print(
        "Event mix -> "
        f"drones={counts.get('drones', 0)}, "
        f"ad_units={counts.get('ad_units', 0)}, "
        f"targets={counts.get('targets', 0)}"
    )
    print("-" * 60)


if __name__ == "__main__":
    main()
