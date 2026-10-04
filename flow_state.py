"""
ChronoGuard v1.0 — Flow State Management (M1)

Implements the FlowTable data structure, 5-tuple canonicalization, stateful
flow aggregation, FIN/RST finalization, and inactivity-timeout eviction.

Data contracts follow spec §5.1 and §5.2 exactly.
"""

from __future__ import annotations

import hashlib
import logging
import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Optional, TypedDict

from scapy.packet import Packet as ScapyPacket

from config import INACTIVITY_TIMEOUT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ADR-006 Option B: Rate-limited debug logging for non-IP packet skips.
# Logs at most once per second, with a count of suppressed messages.
# This reduces log volume ~100× when the Streamlit dashboard generates
# HTTP/WebSocket traffic on lo0 during concurrent operation.
# ---------------------------------------------------------------------------
_SKIP_LOG_INTERVAL: float = 1.0  # seconds
_skip_log_last_ts: float = 0.0
_skip_log_suppressed: int = 0


# ---------------------------------------------------------------------------
# Data Contract: FlowRecord (spec §5.1)
# ---------------------------------------------------------------------------
class FlowRecord(TypedDict):
    """Finalized flow feature record pushed onto the queue.

    Field names and types match spec §5.1 exactly.
    """

    # Identity
    flow_id: str               # deterministic hash of 5-tuple + start timestamp
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int              # IANA protocol number (6=TCP, 17=UDP)

    # Temporal (stateful)
    flow_start_ts: float       # epoch seconds, first packet in flow
    flow_end_ts: float         # epoch seconds, last packet observed
    flow_duration: float       # flow_end_ts - flow_start_ts
    flow_iat_mean: float       # mean inter-arrival time (seconds)
    flow_iat_std: float        # std dev of inter-arrival times (0.0 if <2 packets)

    # Directional byte/packet distribution (stateless aggregates)
    fwd_packet_count: int
    bwd_packet_count: int
    fwd_packet_length_max: int
    fwd_packet_length_mean: float
    bwd_packet_length_max: int
    bwd_packet_length_mean: float
    fwd_packets_per_sec: float
    bwd_packets_per_sec: float

    # Termination
    termination_reason: str    # one of: "FIN", "RST", "TIMEOUT"


# ---------------------------------------------------------------------------
# 5-Tuple Canonicalization (spec §5.2)
# ---------------------------------------------------------------------------
def canonical_key(
    src_ip: str, dst_ip: str, src_port: int, dst_port: int, protocol: int
) -> tuple[str, str, int, int, int]:
    """Produce a direction-independent key for the flow table.

    Sort the two (ip, port) endpoints so that A→B and B→A packets map to the
    same flow entry.  The ORIGINAL (non-canonical) src/dst must still be
    tracked separately per-packet to preserve forward/backward directionality
    (spec §5.2).
    """
    endpoint_a = (src_ip, src_port)
    endpoint_b = (dst_ip, dst_port)
    if endpoint_a <= endpoint_b:
        return (src_ip, dst_ip, src_port, dst_port, protocol)
    return (dst_ip, src_ip, dst_port, src_port, protocol)


def _compute_flow_id(
    src_ip: str, dst_ip: str, src_port: int, dst_port: int,
    protocol: int, start_ts: float,
) -> str:
    """Deterministic hash of 5-tuple + start timestamp (spec §5.1)."""
    raw = f"{src_ip}:{dst_ip}:{src_port}:{dst_port}:{protocol}:{start_ts}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Internal Mutable Flow Entry (not exposed outside this module)
# ---------------------------------------------------------------------------
@dataclass
class _FlowEntry:
    """Mutable state for an in-progress flow tracked by FlowTable."""

    # Canonical key components (direction-independent)
    canon_src_ip: str
    canon_dst_ip: str
    canon_src_port: int
    canon_dst_port: int
    protocol: int

    # Initiator tracking — first packet's original src defines "forward" (spec §5.1)
    initiator_ip: str
    initiator_port: int

    # Timestamps
    start_ts: float
    last_seen_ts: float
    packet_timestamps: list[float] = field(default_factory=list)

    # Forward direction counters
    fwd_packet_count: int = 0
    fwd_packet_lengths: list[int] = field(default_factory=list)

    # Backward direction counters
    bwd_packet_count: int = 0
    bwd_packet_lengths: list[int] = field(default_factory=list)

    def update(self, pkt_ts: float, pkt_len: int, is_forward: bool) -> None:
        """Ingest a single packet's contribution to this flow."""
        self.last_seen_ts = pkt_ts
        self.packet_timestamps.append(pkt_ts)
        if is_forward:
            self.fwd_packet_count += 1
            self.fwd_packet_lengths.append(pkt_len)
        else:
            self.bwd_packet_count += 1
            self.bwd_packet_lengths.append(pkt_len)

    def finalize(self, termination_reason: str) -> FlowRecord:
        """Compute all derived features and produce the immutable FlowRecord."""
        duration = self.last_seen_ts - self.start_ts

        # Inter-arrival times (spec §5.1: 0.0 if <2 packets)
        iats: list[float] = []
        sorted_ts = sorted(self.packet_timestamps)
        for i in range(1, len(sorted_ts)):
            iats.append(sorted_ts[i] - sorted_ts[i - 1])

        iat_mean = statistics.mean(iats) if iats else 0.0
        if len(iats) >= 2:
            iat_std = statistics.stdev(iats)
        else:
            iat_std = 0.0

        # Forward packet stats
        fwd_len_max = max(self.fwd_packet_lengths) if self.fwd_packet_lengths else 0
        fwd_len_mean = (
            statistics.mean(self.fwd_packet_lengths)
            if self.fwd_packet_lengths
            else 0.0
        )

        # Backward packet stats
        bwd_len_max = max(self.bwd_packet_lengths) if self.bwd_packet_lengths else 0
        bwd_len_mean = (
            statistics.mean(self.bwd_packet_lengths)
            if self.bwd_packet_lengths
            else 0.0
        )

        # Packets per second (avoid division by zero)
        if duration > 0.0:
            fwd_pps = self.fwd_packet_count / duration
            bwd_pps = self.bwd_packet_count / duration
        else:
            fwd_pps = float(self.fwd_packet_count)
            bwd_pps = float(self.bwd_packet_count)

        flow_id = _compute_flow_id(
            self.canon_src_ip, self.canon_dst_ip,
            self.canon_src_port, self.canon_dst_port,
            self.protocol, self.start_ts,
        )

        return FlowRecord(
            flow_id=flow_id,
            src_ip=self.canon_src_ip,
            dst_ip=self.canon_dst_ip,
            src_port=self.canon_src_port,
            dst_port=self.canon_dst_port,
            protocol=self.protocol,
            flow_start_ts=self.start_ts,
            flow_end_ts=self.last_seen_ts,
            flow_duration=duration,
            flow_iat_mean=iat_mean,
            flow_iat_std=iat_std,
            fwd_packet_count=self.fwd_packet_count,
            bwd_packet_count=self.bwd_packet_count,
            fwd_packet_length_max=fwd_len_max,
            fwd_packet_length_mean=fwd_len_mean,
            bwd_packet_length_max=bwd_len_max,
            bwd_packet_length_mean=bwd_len_mean,
            fwd_packets_per_sec=fwd_pps,
            bwd_packets_per_sec=bwd_pps,
            termination_reason=termination_reason,
        )


# ---------------------------------------------------------------------------
# FlowTable (spec §5.5)
# ---------------------------------------------------------------------------
class FlowTable:
    """In-memory flow table with 5-tuple canonicalization and eviction.

    Owned exclusively by the sniffer thread — not thread-safe by design
    (single-writer, spec §6).
    """

    def __init__(self, inactivity_timeout: float = INACTIVITY_TIMEOUT) -> None:
        self._flows: dict[tuple[str, str, int, int, int], _FlowEntry] = {}
        self._inactivity_timeout = inactivity_timeout

    @property
    def size(self) -> int:
        """Current number of active (non-finalized) flows in the table."""
        return len(self._flows)

    def ingest_packet(self, pkt: ScapyPacket) -> Optional[FlowRecord]:
        """Update or create flow entry. Returns a finalized FlowRecord
        if this packet triggered finalization (FIN/RST), else None.

        Signature matches spec §5.5.
        """
        from scapy.layers.inet import IP, TCP, UDP

        # --- Extract IP layer (spec §6: skip non-IP silently at debug level) ---
        # ADR-006 Option B: rate-limited logging (at most once per second).
        if not pkt.haslayer(IP):
            global _skip_log_last_ts, _skip_log_suppressed
            now = time.monotonic()
            if (now - _skip_log_last_ts) >= _SKIP_LOG_INTERVAL:
                if _skip_log_suppressed > 0:
                    logger.debug(
                        "Skipping non-IP packet (%d similar messages suppressed)",
                        _skip_log_suppressed,
                    )
                else:
                    logger.debug("Skipping non-IP packet")
                _skip_log_last_ts = now
                _skip_log_suppressed = 0
            else:
                _skip_log_suppressed += 1
            return None

        ip_layer = pkt[IP]
        src_ip: str = ip_layer.src
        dst_ip: str = ip_layer.dst
        protocol: int = ip_layer.proto

        # --- Extract transport ports ---
        if pkt.haslayer(TCP):
            tcp = pkt[TCP]
            src_port = tcp.sport
            dst_port = tcp.dport
        elif pkt.haslayer(UDP):
            udp = pkt[UDP]
            src_port = udp.sport
            dst_port = udp.dport
        else:
            # Non-TCP/UDP IP packet — use port 0 for ICMP, etc.
            src_port = 0
            dst_port = 0

        # --- Timestamp Authority (spec §5.1) ---
        # Use the capture-time wall-clock timestamp provided by scapy's
        # packet `.time` field, which reflects kernel capture time on lo.
        # Do NOT attempt to parse or trust any timestamp encoded inside
        # the replayed payload itself.
        pkt_ts: float = float(pkt.time)

        # Payload length (IP total length, or raw packet length as fallback)
        pkt_len: int = len(ip_layer)

        # --- Canonicalize key (spec §5.2) ---
        key = canonical_key(src_ip, dst_ip, src_port, dst_port, protocol)

        # --- Create or update flow entry ---
        entry = self._flows.get(key)

        if entry is None:
            # New flow — first packet defines the initiator (spec §5.1 Directionality)
            entry = _FlowEntry(
                canon_src_ip=key[0],
                canon_dst_ip=key[1],
                canon_src_port=key[2],
                canon_dst_port=key[3],
                protocol=key[4],
                initiator_ip=src_ip,
                initiator_port=src_port,
                start_ts=pkt_ts,
                last_seen_ts=pkt_ts,
            )
            self._flows[key] = entry

        # Determine directionality relative to the ORIGINAL initiator (spec §5.1)
        is_forward = (src_ip == entry.initiator_ip and src_port == entry.initiator_port)

        entry.update(pkt_ts, pkt_len, is_forward)

        # --- Check for TCP teardown flags (FIN/RST) ---
        if pkt.haslayer(TCP):
            tcp_flags = pkt[TCP].flags
            if tcp_flags is not None:
                flags_int = int(tcp_flags)
                # FIN = 0x01, RST = 0x04
                if flags_int & 0x01:
                    return self._finalize_flow(key, "FIN")
                if flags_int & 0x04:
                    return self._finalize_flow(key, "RST")

        return None

    def evict_stale(self, now: float) -> list[FlowRecord]:
        """Called periodically; returns and removes all flows whose
        last-seen timestamp exceeds the inactivity threshold.

        Signature matches spec §5.5.
        """
        stale_keys: list[tuple[str, str, int, int, int]] = []
        for key, entry in self._flows.items():
            if (now - entry.last_seen_ts) > self._inactivity_timeout:
                stale_keys.append(key)

        finalized: list[FlowRecord] = []
        for key in stale_keys:
            record = self._finalize_flow(key, "TIMEOUT")
            if record is not None:
                finalized.append(record)

        return finalized

    def _finalize_flow(
        self, key: tuple[str, str, int, int, int], reason: str
    ) -> Optional[FlowRecord]:
        """Remove a flow from the table and produce its finalized FlowRecord."""
        entry = self._flows.pop(key, None)
        if entry is None:
            return None

        record = entry.finalize(reason)
        logger.debug(
            "Flow finalized: flow_id=%s reason=%s",
            record["flow_id"],
            record["termination_reason"],
        )
        return record
