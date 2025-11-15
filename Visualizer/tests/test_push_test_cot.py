import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.push_test_cot import SAMPLE_ICON_PACK, build_payloads, parse_args


def test_build_payload_contains_uid_and_callsign():
    args = parse_args([
        "--cot-endpoint",
        "udp://127.0.0.1:6969",
        "--uid",
        "UNIT-123",
        "--callsign",
        "EAGLE",
        "--remarks",
        "Hello TAK",
    ])
    payloads = build_payloads(args)
    assert len(payloads) == 1
    payload = payloads[0].payload.decode("utf-8")
    assert payload.startswith("<?xml version=\"1.0\"")
    assert "UNIT-123" in payload
    assert "EAGLE" in payload
    assert "Hello TAK" in payload


def test_sample_pack_builds_multiple_icons():
    args = parse_args([
        "--cot-endpoint",
        "udp://127.0.0.1:6969",
        "--mode",
        "sample-pack",
    ])
    payloads = build_payloads(args)
    assert len(payloads) == len(SAMPLE_ICON_PACK)
    uids = {payload.uid for payload in payloads}
    assert "ICON-MONUMENT-RED" in uids
    assert "ICON-CAPITOL-TGT" in uids
