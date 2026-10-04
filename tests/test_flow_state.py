"""
Unit tests for flow_state.py — ChronoGuard M1

Covers (per spec §8 deliverables):
  - 5-tuple canonicalization
  - Flow eviction timing
  - Feature vector assembly against FEATURE_ORDER
  - FIN/RST finalization
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from config import FEATURE_ORDER, INACTIVITY_TIMEOUT
from flow_state import FlowRecord, FlowTable, canonical_key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_mock_packet(
    src_ip: str = "10.0.0.1",
    dst_ip: str = "10.0.0.2",
    src_port: int = 12345,
    dst_port: int = 80,
    protocol: int = 6,
    pkt_time: float | None = None,
    pkt_len: int = 100,
    has_tcp: bool = True,
    has_udp: bool = False,
    has_ip: bool = True,
    tcp_flags: int = 0x00,
) -> MagicMock:
    """Create a mock scapy packet with the required layers."""
    pkt = MagicMock()
    pkt.time = pkt_time if pkt_time is not None else time.time()

    if has_ip:
        ip_layer = MagicMock()
        ip_layer.src = src_ip
        ip_layer.dst = dst_ip
        ip_layer.proto = protocol
        ip_layer.__len__ = lambda _self: pkt_len
        pkt.haslayer = lambda layer_cls, _ip=has_ip, _tcp=has_tcp, _udp=has_udp: {
            "IP": _ip, "TCP": _tcp, "UDP": _udp
        }.get(layer_cls.__name__, False)

        # __getitem__ for pkt[IP], pkt[TCP], pkt[UDP]
        tcp_layer = MagicMock()
        tcp_layer.sport = src_port
        tcp_layer.dport = dst_port
        tcp_layer.flags = tcp_flags

        udp_layer = MagicMock()
        udp_layer.sport = src_port
        udp_layer.dport = dst_port

        def getitem(_self, layer_cls):
            if layer_cls.__name__ == "IP":
                return ip_layer
            elif layer_cls.__name__ == "TCP":
                return tcp_layer
            elif layer_cls.__name__ == "UDP":
                return udp_layer
            raise KeyError(layer_cls)

        pkt.__getitem__ = getitem
    else:
        pkt.haslayer = lambda _: False

    return pkt


# ---------------------------------------------------------------------------
# 5-Tuple Canonicalization Tests
# ---------------------------------------------------------------------------
class TestCanonicalKey:
    """Spec §5.2: A→B and B→A produce the same canonical key."""

    def test_same_key_both_directions(self) -> None:
        key_ab = canonical_key("10.0.0.1", "10.0.0.2", 12345, 80, 6)
        key_ba = canonical_key("10.0.0.2", "10.0.0.1", 80, 12345, 6)
        assert key_ab == key_ba

    def test_same_ip_different_ports(self) -> None:
        key_ab = canonical_key("10.0.0.1", "10.0.0.1", 1000, 2000, 6)
        key_ba = canonical_key("10.0.0.1", "10.0.0.1", 2000, 1000, 6)
        assert key_ab == key_ba

    def test_different_protocols_different_keys(self) -> None:
        key_tcp = canonical_key("10.0.0.1", "10.0.0.2", 12345, 80, 6)
        key_udp = canonical_key("10.0.0.1", "10.0.0.2", 12345, 80, 17)
        assert key_tcp != key_udp

    def test_returns_tuple_of_correct_length(self) -> None:
        key = canonical_key("10.0.0.1", "10.0.0.2", 12345, 80, 6)
        assert isinstance(key, tuple)
        assert len(key) == 5


# ---------------------------------------------------------------------------
# Flow Eviction Timing Tests
# ---------------------------------------------------------------------------
class TestFlowEviction:
    """Spec §5.5: flows exceeding 15s inactivity are evicted; younger are not."""

    def test_stale_flows_evicted(self) -> None:
        ft = FlowTable(inactivity_timeout=15.0)

        # Ingest a packet with a timestamp 20s ago
        old_ts = time.time() - 20.0
        pkt = _make_mock_packet(pkt_time=old_ts)
        ft.ingest_packet(pkt)

        assert ft.size == 1
        evicted = ft.evict_stale(time.time())
        assert len(evicted) == 1
        assert evicted[0]["termination_reason"] == "TIMEOUT"
        assert ft.size == 0

    def test_fresh_flows_not_evicted(self) -> None:
        ft = FlowTable(inactivity_timeout=15.0)

        # Ingest a packet with a recent timestamp
        pkt = _make_mock_packet(pkt_time=time.time())
        ft.ingest_packet(pkt)

        assert ft.size == 1
        evicted = ft.evict_stale(time.time())
        assert len(evicted) == 0
        assert ft.size == 1

    def test_exact_boundary_not_evicted(self) -> None:
        """A flow whose last_seen_ts is exactly at the threshold is NOT evicted
        (the condition is strictly > timeout)."""
        ft = FlowTable(inactivity_timeout=15.0)

        now = time.time()
        pkt = _make_mock_packet(pkt_time=now - 15.0)
        ft.ingest_packet(pkt)

        evicted = ft.evict_stale(now)
        assert len(evicted) == 0

    def test_multiple_flows_selective_eviction(self) -> None:
        ft = FlowTable(inactivity_timeout=15.0)

        now = time.time()

        # Old flow
        pkt_old = _make_mock_packet(
            src_ip="10.0.0.1", dst_ip="10.0.0.2", src_port=1000,
            pkt_time=now - 20.0,
        )
        ft.ingest_packet(pkt_old)

        # Fresh flow
        pkt_new = _make_mock_packet(
            src_ip="10.0.0.3", dst_ip="10.0.0.4", src_port=2000,
            pkt_time=now,
        )
        ft.ingest_packet(pkt_new)

        assert ft.size == 2
        evicted = ft.evict_stale(now)
        assert len(evicted) == 1
        assert ft.size == 1


# ---------------------------------------------------------------------------
# Feature Vector Assembly Tests
# ---------------------------------------------------------------------------
class TestFeatureVectorAssembly:
    """Spec §5.3: FlowRecord must contain all fields referenced by FEATURE_ORDER."""

    def _make_finalized_record(self) -> FlowRecord:
        ft = FlowTable()
        now = time.time()

        # Send 3 forward packets
        for i in range(3):
            pkt = _make_mock_packet(
                src_ip="10.0.0.1", dst_ip="10.0.0.2",
                src_port=12345, dst_port=80,
                pkt_time=now + i * 0.1, pkt_len=100 + i * 10,
            )
            result = ft.ingest_packet(pkt)

        # Send a FIN to finalize
        pkt_fin = _make_mock_packet(
            src_ip="10.0.0.1", dst_ip="10.0.0.2",
            src_port=12345, dst_port=80,
            pkt_time=now + 0.4, tcp_flags=0x01,
        )
        record = ft.ingest_packet(pkt_fin)
        assert record is not None
        return record

    def test_all_feature_order_fields_present(self) -> None:
        """Every field in FEATURE_ORDER must exist in the FlowRecord."""
        record = self._make_finalized_record()
        for feat in FEATURE_ORDER:
            assert feat in record, f"Missing feature: {feat}"

    def test_feature_vector_length(self) -> None:
        """Feature vector extracted per FEATURE_ORDER has correct length."""
        record = self._make_finalized_record()
        features = [float(record[feat]) for feat in FEATURE_ORDER]
        assert len(features) == len(FEATURE_ORDER)

    def test_all_features_are_numeric(self) -> None:
        """All FEATURE_ORDER fields must be numeric (int or float)."""
        record = self._make_finalized_record()
        for feat in FEATURE_ORDER:
            val = record[feat]
            assert isinstance(val, (int, float)), (
                f"Feature '{feat}' has non-numeric value: {val!r} ({type(val)})"
            )

    def test_flow_record_has_all_spec_fields(self) -> None:
        """FlowRecord has exactly the fields defined in spec §5.1."""
        record = self._make_finalized_record()
        expected_fields = {
            "flow_id", "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
            "flow_start_ts", "flow_end_ts", "flow_duration",
            "flow_iat_mean", "flow_iat_std",
            "fwd_packet_count", "bwd_packet_count",
            "fwd_packet_length_max", "fwd_packet_length_mean",
            "bwd_packet_length_max", "bwd_packet_length_mean",
            "fwd_packets_per_sec", "bwd_packets_per_sec",
            "termination_reason",
        }
        assert set(record.keys()) == expected_fields


# ---------------------------------------------------------------------------
# FIN/RST Finalization Tests
# ---------------------------------------------------------------------------
class TestFinalization:
    """Spec §5.5: FIN/RST triggers finalization and returns a FlowRecord."""

    def test_fin_triggers_finalization(self) -> None:
        ft = FlowTable()
        pkt = _make_mock_packet(tcp_flags=0x01)  # FIN
        record = ft.ingest_packet(pkt)
        assert record is not None
        assert record["termination_reason"] == "FIN"
        assert ft.size == 0

    def test_rst_triggers_finalization(self) -> None:
        ft = FlowTable()
        pkt = _make_mock_packet(tcp_flags=0x04)  # RST
        record = ft.ingest_packet(pkt)
        assert record is not None
        assert record["termination_reason"] == "RST"
        assert ft.size == 0

    def test_normal_packet_no_finalization(self) -> None:
        ft = FlowTable()
        pkt = _make_mock_packet(tcp_flags=0x10)  # ACK only
        record = ft.ingest_packet(pkt)
        assert record is None
        assert ft.size == 1

    def test_non_ip_packet_skipped(self) -> None:
        ft = FlowTable()
        pkt = _make_mock_packet(has_ip=False)
        record = ft.ingest_packet(pkt)
        assert record is None
        assert ft.size == 0

    def test_bidirectional_flow_tracking(self) -> None:
        """Forward and backward packets in the same flow are tracked together."""
        ft = FlowTable()
        now = time.time()

        # Forward: A → B
        pkt_fwd = _make_mock_packet(
            src_ip="10.0.0.1", dst_ip="10.0.0.2",
            src_port=12345, dst_port=80,
            pkt_time=now,
        )
        ft.ingest_packet(pkt_fwd)

        # Backward: B → A (same canonical flow)
        pkt_bwd = _make_mock_packet(
            src_ip="10.0.0.2", dst_ip="10.0.0.1",
            src_port=80, dst_port=12345,
            pkt_time=now + 0.1,
        )
        ft.ingest_packet(pkt_bwd)

        assert ft.size == 1  # Same flow

        # FIN from A → B
        pkt_fin = _make_mock_packet(
            src_ip="10.0.0.1", dst_ip="10.0.0.2",
            src_port=12345, dst_port=80,
            pkt_time=now + 0.2, tcp_flags=0x01,
        )
        record = ft.ingest_packet(pkt_fin)
        assert record is not None
        assert record["fwd_packet_count"] == 2  # pkt_fwd + pkt_fin
        assert record["bwd_packet_count"] == 1  # pkt_bwd

    def test_flow_id_is_deterministic(self) -> None:
        """Same 5-tuple + start timestamp → same flow_id."""
        ft1 = FlowTable()
        ft2 = FlowTable()
        ts = 1700000000.0

        pkt1 = _make_mock_packet(pkt_time=ts, tcp_flags=0x01)
        pkt2 = _make_mock_packet(pkt_time=ts, tcp_flags=0x01)

        rec1 = ft1.ingest_packet(pkt1)
        rec2 = ft2.ingest_packet(pkt2)

        assert rec1 is not None and rec2 is not None
        assert rec1["flow_id"] == rec2["flow_id"]

    def test_iat_with_single_packet(self) -> None:
        """IAT mean/std should be 0.0 for a single-packet flow (spec §5.1)."""
        ft = FlowTable()
        pkt = _make_mock_packet(tcp_flags=0x01)  # FIN immediately
        record = ft.ingest_packet(pkt)
        assert record is not None
        assert record["flow_iat_mean"] == 0.0
        assert record["flow_iat_std"] == 0.0
