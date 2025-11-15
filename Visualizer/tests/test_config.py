import pytest

from swarm_visualizer.config import PyTakRuntimeConfig


def test_cot_endpoint_rejects_http_scheme():
    with pytest.raises(ValueError):
        PyTakRuntimeConfig(cot_endpoint="http://127.0.0.1:6969")


def test_cot_endpoint_accepts_udp_scheme():
    cfg = PyTakRuntimeConfig(cot_endpoint="udp://10.0.0.1:4242")
    assert cfg.cot_endpoint == "udp://10.0.0.1:4242"
