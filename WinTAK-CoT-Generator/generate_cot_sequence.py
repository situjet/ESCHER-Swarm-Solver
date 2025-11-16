"""Generate and push CoT sequences to WinTAK for Swarm Defense visualization."""

from __future__ import annotations

import argparse
import math
import socket
import sys
import time
from pathlib import Path
from typing import List, Optional, Set, Tuple

from cot_helpers import (
    COT_TYPES,
    TOT_COLORS,
    arena_to_latlon,
    format_cot_event,
    format_delete_event,
    wrap_xml,
)
from game_simulator import GameSnapshot, generate_game_sequence

DEFAULT_AD_FLASH_PERIOD = 0.4  # seconds
AD_FOV_HALF_ANGLE = math.radians(35.0)
AD_FOV_SEGMENTS = 12
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT_PATH = PROJECT_ROOT / "Visualizer" / "swarm_large_snapshot.json"


def _build_ad_fov_detail(
    ad_unit,
) -> str:
    """Return a TAK spatialEvent fan representing the AD's field of view."""

    radius = max(float(ad_unit.coverage_radius), 0.0)
    if radius <= 0.0:
        return ""
    center_lat, center_lon = arena_to_latlon(ad_unit.position[0], ad_unit.position[1])
    start_angle = ad_unit.orientation - AD_FOV_HALF_ANGLE
    end_angle = ad_unit.orientation + AD_FOV_HALF_ANGLE
    samples = max(AD_FOV_SEGMENTS, 1)
    points = [(center_lat, center_lon)]
    for idx in range(samples + 1):
        angle = start_angle + (end_angle - start_angle) * (idx / samples)
        row = ad_unit.position[0] + math.sin(angle) * radius
        col = ad_unit.position[1] + math.cos(angle) * radius
        lat, lon = arena_to_latlon(row, col)
        points.append((lat, lon))
    points.append((center_lat, center_lon))
    point_xml = "".join(
        f'<point lat="{lat:.6f}" lon="{lon:.6f}" hae="0" ce="0" le="0"/>'
        for lat, lon in points
    )
    return (
        "<spatialEvent><shape>polygon</shape>"
        '<strokeColor argb="-16738193"/><strokeWeight>2</strokeWeight>'
        '<fillColor argb="872482815"/>'
        f"<polygon>{point_xml}</polygon></spatialEvent>"
    )


def send_tcp(host: str, port: int, payload: bytes) -> None:
    """Send CoT payload via TCP."""
    try:
        with socket.create_connection((host, port), timeout=5) as sock:
            sock.sendall(payload)
    except Exception as e:
        print(f"Warning: Failed to send TCP payload: {e}")


def _emit_event_xml(event_xml: str, host: str, port: int, dry_run: bool, label: str) -> None:
    """Send or preview a single CoT event."""

    payload = wrap_xml(event_xml)
    if dry_run:
        print(f"\n[{label}] Would send {len(payload)} bytes:")
        print(event_xml[:200] + "..." if len(event_xml) > 200 else event_xml)
    else:
        send_tcp(host, port, payload)


def snapshot_to_cot_events(
    snapshot: GameSnapshot,
    prefix: str = "SWARM",
    flash_period: float = DEFAULT_AD_FLASH_PERIOD,
) -> Tuple[List[str], Set[str]]:
    """
    Convert a game snapshot to CoT events and their active UIDs.
    
    Args:
        snapshot: GameSnapshot object
        prefix: Prefix for UIDs and callsigns
    
    Returns:
        Tuple of (CoT XML strings, active UIDs for this frame)
    """
    events: List[str] = []
    active_uids: Set[str] = set()
    # Generate drone events
    for drone in snapshot.drones:
        if not drone.visible:
            continue
        lat, lon = arena_to_latlon(drone.position[0], drone.position[1])
        cot_type = COT_TYPES["drone_active"]
        
        tot_color = TOT_COLORS.get(drone.tot_offset, "UNKNOWN")
        remarks = f"Drone {drone.drone_id} -> Target {drone.target_idx}, TOT+{drone.tot_offset}s ({tot_color}), status={drone.status}"
        if drone.destroyed_by:
            remarks += f" | threat: {drone.destroyed_by}"
        
        uid = f"{prefix}-DRONE-{drone.drone_id}"
        events.append(
            format_cot_event(
                uid=uid,
                callsign=f"{prefix}-D{drone.drone_id}",
                cot_type=cot_type,
                lat=lat,
                lon=lon,
                hae=100,
                ce=15,
                le=15,
                remarks=remarks,
            )
        )
        active_uids.add(uid)
    
    # Generate interceptor events (only while visible)
    for interceptor in snapshot.interceptors:
        if not interceptor.visible:
            continue
        lat, lon = arena_to_latlon(interceptor.position[0], interceptor.position[1])
        remarks = f"Interceptor {interceptor.interceptor_id}"
        if interceptor.assigned_drone is not None:
            remarks += f", tracking Drone {interceptor.assigned_drone}"
        else:
            remarks += ", no target"
        
        uid = f"{prefix}-INT-{interceptor.interceptor_id}"
        events.append(
            format_cot_event(
                uid=uid,
                callsign=f"{prefix}-I{interceptor.interceptor_id}",
                cot_type=COT_TYPES["interceptor"],
                lat=lat,
                lon=lon,
                hae=150,
                ce=20,
                le=20,
                remarks=remarks,
            )
        )
        active_uids.add(uid)
    
    # Generate AD unit events
    for ad_unit in snapshot.ad_units:
        if not ad_unit.alive:
            continue
        lat, lon = arena_to_latlon(ad_unit.position[0], ad_unit.position[1])
        status = "ACTIVE"
        remarks = (
            f"AD Unit {ad_unit.ad_id}, coverage={ad_unit.coverage_radius:.1f}m, {status}"
        )
        if ad_unit.engaged_drones:
            remarks += f" | engaging: {', '.join(f'D{d}' for d in ad_unit.engaged_drones)}"
        
        uid = f"{prefix}-AD-{ad_unit.ad_id}"
        events.append(
            format_cot_event(
                uid=uid,
                callsign=f"{prefix}-AD{ad_unit.ad_id}",
                cot_type=COT_TYPES["ad_unit"],
                lat=lat,
                lon=lon,
                hae=50,
                ce=25,
                le=25,
                remarks=remarks,
            )
        )
        active_uids.add(uid)
        fov_detail = _build_ad_fov_detail(ad_unit)
        if fov_detail:
            fov_uid = f"{prefix}-ADFOV-{ad_unit.ad_id}"
            events.append(
                format_cot_event(
                    uid=fov_uid,
                    callsign=f"{prefix}-FOV-{ad_unit.ad_id}",
                    cot_type=COT_TYPES["ad_fov"],
                    lat=lat,
                    lon=lon,
                    hae=25,
                    ce=int(ad_unit.coverage_radius * 2),
                    le=int(ad_unit.coverage_radius * 2),
                    remarks=f"AD {ad_unit.ad_id} field of view",
                    stale_seconds=max(flash_period, 0.5),
                    extra_detail=fov_detail,
                )
            )
            active_uids.add(fov_uid)
    
    # Generate target events (only while active)
    for target in snapshot.targets:
        if not target.visible:
            continue
        lat, lon = arena_to_latlon(target.position[0], target.position[1])
        remarks = f"Target {target.target_id}, value={target.value:.0f}, ACTIVE"
        
        uid = f"{prefix}-TGT-{target.target_id}"
        events.append(
            format_cot_event(
                uid=uid,
                callsign=f"{prefix}-T{target.target_id}",
                cot_type=COT_TYPES["target"],
                lat=lat,
                lon=lon,
                hae=0,
                ce=20,
                le=20,
                remarks=remarks,
            )
        )
        active_uids.add(uid)
    
    
    return events, active_uids


def push_snapshot_to_wintak(
    snapshot: GameSnapshot,
    host: str,
    port: int,
    prefix: str = "SWARM",
    dry_run: bool = False,
    previous_active_uids: Optional[Set[str]] = None,
    flash_period: float = DEFAULT_AD_FLASH_PERIOD,
) -> Tuple[int, Set[str]]:
    """
    Push a single game snapshot to WinTAK as CoT events.
    
    Args:
        snapshot: GameSnapshot to visualize
        host: TCP host
        port: TCP port
        prefix: UID/callsign prefix
        dry_run: If True, print instead of sending
        previous_active_uids: Active UIDs from the prior frame for cleanup
    
    Returns:
        (Number of events sent, active UIDs for this snapshot)
    """
    events, active_uids = snapshot_to_cot_events(snapshot, prefix, flash_period=flash_period)
    deletion_events: List[str] = []
    label_prefix = f"{'DRY RUN' if dry_run else 'LIVE'} - t={snapshot.time:.2f}s"

    if previous_active_uids is not None:
        stale_uids = previous_active_uids - active_uids
        for uid in sorted(stale_uids):
            deletion_events.append(format_delete_event(uid, remarks="Auto-delete"))
    
    total_events = len(events) + len(deletion_events)

    for event_xml in deletion_events:
        _emit_event_xml(event_xml, host, port, dry_run, f"{label_prefix} DELETE")
    for event_xml in events:
        _emit_event_xml(event_xml, host, port, dry_run, label_prefix)
    
    return total_events, active_uids


def push_sequence_to_wintak(
    snapshots: List[GameSnapshot],
    host: str,
    port: int,
    interval: float = 0.5,
    prefix: str = "SWARM",
    dry_run: bool = False,
    flash_period: float = DEFAULT_AD_FLASH_PERIOD,
) -> None:
    """
    Push a complete game sequence to WinTAK.
    
    Args:
        snapshots: List of GameSnapshot objects
        host: TCP host
        port: TCP port
        interval: Seconds between snapshots
        prefix: UID/callsign prefix
        dry_run: If True, print instead of sending
    """
    print(f"Pushing {len(snapshots)} snapshots to {host}:{port}")
    print(f"Time range: {snapshots[0].time:.2f}s to {snapshots[-1].time:.2f}s")
    print(f"Interval: {interval}s between updates")
    print(f"Total duration: {len(snapshots) * interval:.1f}s")
    print()
    
    previous_active: Set[str] = set()
    for idx, snapshot in enumerate(snapshots):
        event_count, current_active = push_snapshot_to_wintak(
            snapshot,
            host,
            port,
            prefix=prefix,
            dry_run=dry_run,
            previous_active_uids=previous_active,
            flash_period=flash_period,
        )
        previous_active = current_active
        
        if not dry_run:
            print(f"[{idx+1}/{len(snapshots)}] t={snapshot.time:.2f}s: Sent {event_count} events")
        
        # Sleep between snapshots (except for the last one)
        if idx < len(snapshots) - 1:
            time.sleep(interval)
    
    if previous_active:
        cleanup_label = "DRY RUN - cleanup" if dry_run else "LIVE - cleanup"
        for uid in sorted(previous_active):
            delete_xml = format_delete_event(uid, remarks="Sequence complete")
            _emit_event_xml(delete_xml, host, port, dry_run, cleanup_label)
    
    print("\nSequence complete!")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate and push Swarm Defense CoT sequences to WinTAK",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate and push a game with default settings
  python generate_cot_sequence.py
  
  # Use a specific seed and faster updates
  python generate_cot_sequence.py --seed 42 --interval 0.25
  
  # Dry run to see what would be sent
  python generate_cot_sequence.py --dry-run
  
  # Use custom endpoint
  python generate_cot_sequence.py --host 192.168.1.100 --port 8087
        """
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for game generation (default: random)",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=DEFAULT_SNAPSHOT_PATH,
        help=(
            "Path to the Swarm-AD-Large snapshot JSON exported by demo_animation.py "
            f"(default: {DEFAULT_SNAPSHOT_PATH})"
        ),
    )
    parser.add_argument(
        "--fresh-run",
        action="store_true",
        help="Ignore any snapshot file and generate a brand-new simulated episode",
    )
    parser.add_argument(
        "--time-step",
        type=float,
        default=0.25,
        help="Time step between game snapshots in seconds (default: 0.25)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Interval between CoT updates in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--fps-multiplier",
        type=float,
        default=2.0,
        help="Multiplier to increase CoT update frequency (default: 2.0)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="TCP host for WinTAK (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6969,
        help="TCP port for WinTAK (default: 6969)",
    )
    parser.add_argument(
        "--prefix",
        default="SWARM",
        help="UID/callsign prefix (default: SWARM)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print CoT events instead of sending them",
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Generate seed if not provided
    if args.seed is None:
        import random
        args.seed = random.randint(0, 2**31 - 1)
    
    print("=" * 60)
    print("Swarm Defense WinTAK CoT Generator")
    print("=" * 60)
    print(f"Seed: {args.seed}")
    fps_multiplier = max(args.fps_multiplier, 1.0)
    effective_time_step = max(args.time_step / fps_multiplier, 0.05)
    effective_interval = max(args.interval / fps_multiplier, 0.05)
    effective_flash_period = max(DEFAULT_AD_FLASH_PERIOD / fps_multiplier, 0.2)

    snapshot_path = None if args.fresh_run else args.snapshot
    print(f"Time step: {args.time_step}s (effective {effective_time_step:.3f}s @ {fps_multiplier:.1f}x fps)")
    print(f"Update interval: {args.interval}s (effective {effective_interval:.3f}s)")
    print(f"Target: tcp://{args.host}:{args.port}")
    print(f"Prefix: {args.prefix}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    if snapshot_path is None:
        print("Snapshot source: disabled (fresh run)")
    else:
        print(f"Snapshot source: {snapshot_path}")
    print("=" * 60)
    print()
    
    try:
        # Generate game sequence
        print("Generating game sequence...")
        snapshots = generate_game_sequence(args.seed, effective_time_step, snapshot_path=snapshot_path)
        print(f"Generated {len(snapshots)} snapshots")
        print()
        
        # Push to WinTAK
        push_sequence_to_wintak(
            snapshots,
            args.host,
            args.port,
            interval=effective_interval,
            prefix=args.prefix,
            dry_run=args.dry_run,
            flash_period=effective_flash_period,
        )
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
