"""Generate and push CoT sequences to WinTAK for Swarm Defense visualization."""

from __future__ import annotations

import argparse
import socket
import sys
import time
from typing import List

from cot_helpers import (
    COT_TYPES,
    TOT_COLORS,
    arena_to_latlon,
    format_cot_event,
    wrap_xml,
)
from game_simulator import GameSnapshot, generate_game_sequence

AD_FLASH_PERIOD = 0.4  # seconds


def send_tcp(host: str, port: int, payload: bytes) -> None:
    """Send CoT payload via TCP."""
    try:
        with socket.create_connection((host, port), timeout=5) as sock:
            sock.sendall(payload)
    except Exception as e:
        print(f"Warning: Failed to send TCP payload: {e}")


def snapshot_to_cot_events(snapshot: GameSnapshot, prefix: str = "SWARM") -> List[str]:
    """
    Convert a game snapshot to a list of CoT event XML strings.
    
    Args:
        snapshot: GameSnapshot object
        prefix: Prefix for UIDs and callsigns
    
    Returns:
        List of CoT XML event strings
    """
    events = []
    flash_on = int(snapshot.time / AD_FLASH_PERIOD) % 2 == 0
    
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
        
        events.append(
            format_cot_event(
                uid=f"{prefix}-DRONE-{drone.drone_id}",
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
        
        events.append(
            format_cot_event(
                uid=f"{prefix}-INT-{interceptor.interceptor_id}",
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
    
    # Generate AD unit events
    for ad_unit in snapshot.ad_units:
        if not ad_unit.alive:
            continue
        lat, lon = arena_to_latlon(ad_unit.position[0], ad_unit.position[1])
        remarks = f"AD Unit {ad_unit.ad_id}, coverage={ad_unit.coverage_radius:.1f}m"
        if ad_unit.engaged_drones:
            remarks += f", engaging: {', '.join(f'D{d}' for d in ad_unit.engaged_drones)}"
        else:
            remarks += ", idle"
        
        events.append(
            format_cot_event(
                uid=f"{prefix}-AD-{ad_unit.ad_id}",
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
    
    # Generate target events (only while active)
    for target in snapshot.targets:
        if not target.visible:
            continue
        lat, lon = arena_to_latlon(target.position[0], target.position[1])
        remarks = f"Target {target.target_id}, value={target.value:.0f}, ACTIVE"
        
        events.append(
            format_cot_event(
                uid=f"{prefix}-TGT-{target.target_id}",
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
    
    # Flashing AD engagement connectors
    if flash_on:
        for engagement in snapshot.engagements:
            mid_row = (engagement.ad_position[0] + engagement.drone_position[0]) / 2.0
            mid_col = (engagement.ad_position[1] + engagement.drone_position[1]) / 2.0
            lat, lon = arena_to_latlon(mid_row, mid_col)
            remarks = (
                f"AD {engagement.ad_id} engaging Drone {engagement.drone_id}"
                f" | intercept @ {engagement.intercept_time:.1f}s"
            )
            events.append(
                format_cot_event(
                    uid=f"{prefix}-ADLINK-{engagement.ad_id}-{engagement.drone_id}",
                    callsign=f"{prefix}-LINK-{engagement.ad_id}-{engagement.drone_id}",
                    cot_type=COT_TYPES["ad_engagement"],
                    lat=lat,
                    lon=lon,
                    hae=25,
                    ce=10,
                    le=10,
                    remarks=remarks,
                )
            )
    
    return events


def push_snapshot_to_wintak(
    snapshot: GameSnapshot,
    host: str,
    port: int,
    prefix: str = "SWARM",
    dry_run: bool = False,
) -> int:
    """
    Push a single game snapshot to WinTAK as CoT events.
    
    Args:
        snapshot: GameSnapshot to visualize
        host: TCP host
        port: TCP port
        prefix: UID/callsign prefix
        dry_run: If True, print instead of sending
    
    Returns:
        Number of events sent
    """
    events = snapshot_to_cot_events(snapshot, prefix)
    
    for event_xml in events:
        payload = wrap_xml(event_xml)
        
        if dry_run:
            print(f"\n[DRY RUN - t={snapshot.time:.2f}s] Would send {len(payload)} bytes:")
            print(event_xml[:200] + "..." if len(event_xml) > 200 else event_xml)
        else:
            send_tcp(host, port, payload)
    
    return len(events)


def push_sequence_to_wintak(
    snapshots: List[GameSnapshot],
    host: str,
    port: int,
    interval: float = 0.5,
    prefix: str = "SWARM",
    dry_run: bool = False,
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
    
    for idx, snapshot in enumerate(snapshots):
        event_count = push_snapshot_to_wintak(
            snapshot,
            host,
            port,
            prefix=prefix,
            dry_run=dry_run,
        )
        
        if not dry_run:
            print(f"[{idx+1}/{len(snapshots)}] t={snapshot.time:.2f}s: Sent {event_count} events")
        
        # Sleep between snapshots (except for the last one)
        if idx < len(snapshots) - 1:
            time.sleep(interval)
    
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
    print(f"Time step: {args.time_step}s")
    print(f"Update interval: {args.interval}s")
    print(f"Target: tcp://{args.host}:{args.port}")
    print(f"Prefix: {args.prefix}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print("=" * 60)
    print()
    
    try:
        # Generate game sequence
        print("Generating game sequence...")
        snapshots = generate_game_sequence(args.seed, args.time_step)
        print(f"Generated {len(snapshots)} snapshots")
        print()
        
        # Push to WinTAK
        push_sequence_to_wintak(
            snapshots,
            args.host,
            args.port,
            interval=args.interval,
            prefix=args.prefix,
            dry_run=args.dry_run,
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
