"""Simple test to verify TCP connection to WinTAK."""

import socket
import sys
from cot_helpers import format_cot_event, wrap_xml, PITTSBURGH_AIRPORT_LAT, PITTSBURGH_AIRPORT_LON


def test_connection(host: str = "127.0.0.1", port: int = 6969) -> bool:
    """Test TCP connection to WinTAK endpoint."""
    print(f"Testing connection to {host}:{port}...")
    
    try:
        # Create a simple test event at Pittsburgh Airport
        event = format_cot_event(
            uid="TEST-CONNECTION",
            callsign="TEST",
            cot_type="a-f-A-M-F",
            lat=PITTSBURGH_AIRPORT_LAT,
            lon=PITTSBURGH_AIRPORT_LON,
            hae=100,
            ce=20,
            le=20,
            remarks="Connection test from Swarm CoT Generator",
        )
        
        payload = wrap_xml(event)
        print(f"Sending {len(payload)} bytes...")
        
        with socket.create_connection((host, port), timeout=5) as sock:
            sock.sendall(payload)
            print("✓ Successfully sent CoT event!")
            return True
            
    except ConnectionRefusedError:
        print(f"✗ Connection refused. Is WinTAK listening on {host}:{port}?")
        return False
    except socket.timeout:
        print(f"✗ Connection timeout. Check if {host}:{port} is accessible.")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


if __name__ == "__main__":
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 6969
    
    success = test_connection(host, port)
    sys.exit(0 if success else 1)
