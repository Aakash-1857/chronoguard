"""
Unit tests for sniffer.py — ChronoGuard M1

Covers (per spec §8 deliverables):
  - Mock packet processing through FlowTable.ingest_packet
  - Non-IP packet skip behavior (spec §6)
"""

from __future__ import annotations

import queue
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from flow_state import FlowRecord, FlowTable


# ---------------------------------------------------------------------------
# Helpers (same mock factory as test_flow_state)
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
# Packet Processing via FlowTable Tests
# ---------------------------------------------------------------------------
class TestPacketProcessing:
    """Verify that packets are correctly processed through FlowTable."""

    def test_tcp_packet_creates_flow(self) -> None:
        """A TCP packet should create a flow entry in the flow table."""
        ft = FlowTable()
        pkt = _make_mock_packet()
        result = ft.ingest_packet(pkt)
        assert result is None  # No finalization yet
        assert ft.size == 1

    def test_udp_packet_creates_flow(self) -> None:
        """A UDP packet should create a flow entry."""
        ft = FlowTable()
        pkt = _make_mock_packet(
            protocol=17, has_tcp=False, has_udp=True,
        )
        result = ft.ingest_packet(pkt)
        assert result is None
        assert ft.size == 1

    def test_multiple_packets_same_flow(self) -> None:
        """Multiple packets with same 5-tuple → single flow."""
        ft = FlowTable()
        now = time.time()

        for i in range(5):
            pkt = _make_mock_packet(pkt_time=now + i * 0.1)
            ft.ingest_packet(pkt)

        assert ft.size == 1

    def test_different_5tuples_different_flows(self) -> None:
        """Packets with different 5-tuples → separate flows."""
        ft = FlowTable()

        pkt1 = _make_mock_packet(src_port=1000)
        pkt2 = _make_mock_packet(src_port=2000)

        ft.ingest_packet(pkt1)
        ft.ingest_packet(pkt2)

        assert ft.size == 2

    def test_finalized_record_has_correct_type(self) -> None:
        """Finalized flow should be a dict matching FlowRecord structure."""
        ft = FlowTable()
        pkt = _make_mock_packet(tcp_flags=0x01)  # FIN
        record = ft.ingest_packet(pkt)

        assert record is not None
        assert isinstance(record, dict)
        assert "flow_id" in record
        assert "termination_reason" in record
        assert record["termination_reason"] == "FIN"


# ---------------------------------------------------------------------------
# Non-IP Packet Skip Tests
# ---------------------------------------------------------------------------
class TestNonIPSkip:
    """Spec §6: malformed/non-IP packets — skip silently at debug level."""

    def test_non_ip_packet_returns_none(self) -> None:
        """Non-IP packets should be skipped (return None, no flow created)."""
        ft = FlowTable()
        pkt = _make_mock_packet(has_ip=False)
        result = ft.ingest_packet(pkt)
        assert result is None
        assert ft.size == 0

    def test_non_ip_packet_after_ip_packet(self) -> None:
        """Non-IP packet after an IP packet should not affect existing flows."""
        ft = FlowTable()

        # First, a valid IP packet
        pkt_ip = _make_mock_packet()
        ft.ingest_packet(pkt_ip)
        assert ft.size == 1

        # Then, a non-IP packet
        pkt_non_ip = _make_mock_packet(has_ip=False)
        result = ft.ingest_packet(pkt_non_ip)
        assert result is None
        assert ft.size == 1  # Unchanged


# ---------------------------------------------------------------------------
# Queue Integration Test (lightweight, no actual sniffing)
# ---------------------------------------------------------------------------
class TestQueueIntegration:
    """Verify that finalized records are correctly pushed to a queue."""

    def test_finalized_record_enqueued(self) -> None:
        """When a FIN packet finalizes a flow, the record should be
        pushable to a queue (simulating what run_sniffer does)."""
        ft = FlowTable()
        out_queue: queue.Queue[FlowRecord] = queue.Queue(maxsize=100)

        pkt = _make_mock_packet(tcp_flags=0x01)  # FIN
        record = ft.ingest_packet(pkt)

        assert record is not None
        out_queue.put_nowait(record)
        assert out_queue.qsize() == 1

        dequeued = out_queue.get_nowait()
        assert dequeued["flow_id"] == record["flow_id"]
