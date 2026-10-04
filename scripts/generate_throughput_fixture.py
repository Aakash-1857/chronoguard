"""
Throughput Fixture PCAP Generator — ChronoGuard

Generates a synthetic PCAP with ~6,000 packets across 60+ distinct 5-tuples
(50 TCP + 10 UDP) for use in M1 throughput acceptance testing.

TCP flows are FIN-terminated (FIN → FIN-ACK → ACK) to exercise the sniffer's
FIN-based finalization path, not just idle-timeout eviction.  Payload sizes
are varied per-flow to produce realistic per-flow feature distributions.

The output uses BSD loopback link-layer (linktype 0 / DLT_NULL) so it can
be replayed directly on macOS lo0 via tcpreplay.

Usage:
    python scripts/generate_throughput_fixture.py

Output:
    tests/fixtures/throughput_fixture.pcap
"""

from __future__ import annotations

import os
import random
import sys
import time

# ---------------------------------------------------------------------------
# Ensure scapy is importable
# ---------------------------------------------------------------------------
try:
    from scapy.all import IP, TCP, UDP, Raw, Ether, wrpcap, conf
    from scapy.layers.l2 import Loopback
except ImportError:
    print("ERROR: scapy is required.  pip install scapy", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "throughput_fixture.pcap")

NUM_TCP_FLOWS = 50
NUM_UDP_FLOWS = 10
# Target ~6,000 packets total.  With 50 TCP flows (each getting data + FIN
# teardown) and 10 UDP flows, we distribute packets across flows.
# TCP: 50 flows × ~100 data packets = 5,000  + 3 teardown packets each = 5,150
# UDP: 10 flows × ~85 packets = 850
# Total ≈ 6,000
TCP_DATA_PACKETS_PER_FLOW = 100
UDP_PACKETS_PER_FLOW = 85

SEED = 42


def _make_tcp_flow(
    flow_idx: int,
    rng: random.Random,
    base_ts: float,
) -> list:
    """Generate a single TCP flow with data packets and FIN teardown."""
    src_port = 10000 + flow_idx * 2
    dst_port = 10001 + flow_idx * 2
    # Use distinct IPs within 10.0.0.0/8 to create unique 5-tuples
    src_ip = f"10.0.{flow_idx // 256}.{flow_idx % 256}"
    dst_ip = f"10.1.{flow_idx // 256}.{flow_idx % 256}"

    payload_size = rng.randint(40, 512)
    packets = []

    # Stagger flow start times so they overlap realistically
    flow_start = base_ts + flow_idx * 0.002  # 2ms between flow starts

    # --- Data packets (forward direction) ---
    for i in range(TCP_DATA_PACKETS_PER_FLOW):
        ts = flow_start + i * 0.001  # 1ms inter-arrival
        pkt = (
            Loopback(type=2)  # AF_INET on BSD
            / IP(src=src_ip, dst=dst_ip)
            / TCP(sport=src_port, dport=dst_port, flags="PA", seq=1000 + i * payload_size, ack=1)
            / Raw(load=bytes(rng.getrandbits(8) for _ in range(payload_size)))
        )
        pkt.time = ts
        packets.append(pkt)

    # --- FIN teardown (3-packet: FIN, FIN-ACK, ACK) ---
    teardown_ts = flow_start + TCP_DATA_PACKETS_PER_FLOW * 0.001

    fin_pkt = (
        Loopback(type=2)
        / IP(src=src_ip, dst=dst_ip)
        / TCP(sport=src_port, dport=dst_port, flags="FA", seq=1000 + TCP_DATA_PACKETS_PER_FLOW * payload_size, ack=1)
    )
    fin_pkt.time = teardown_ts
    packets.append(fin_pkt)

    fin_ack_pkt = (
        Loopback(type=2)
        / IP(src=dst_ip, dst=src_ip)
        / TCP(sport=dst_port, dport=src_port, flags="FA", seq=1, ack=1000 + TCP_DATA_PACKETS_PER_FLOW * payload_size + 1)
    )
    fin_ack_pkt.time = teardown_ts + 0.0001
    packets.append(fin_ack_pkt)

    ack_pkt = (
        Loopback(type=2)
        / IP(src=src_ip, dst=dst_ip)
        / TCP(sport=src_port, dport=dst_port, flags="A", seq=1000 + TCP_DATA_PACKETS_PER_FLOW * payload_size + 1, ack=2)
    )
    ack_pkt.time = teardown_ts + 0.0002
    packets.append(ack_pkt)

    return packets


def _make_udp_flow(
    flow_idx: int,
    rng: random.Random,
    base_ts: float,
) -> list:
    """Generate a single UDP flow (no teardown — relies on idle-timeout)."""
    src_port = 20000 + flow_idx * 2
    dst_port = 20001 + flow_idx * 2
    src_ip = f"10.2.{flow_idx // 256}.{flow_idx % 256}"
    dst_ip = f"10.3.{flow_idx // 256}.{flow_idx % 256}"

    payload_size = rng.randint(64, 256)
    packets = []

    flow_start = base_ts + flow_idx * 0.003

    for i in range(UDP_PACKETS_PER_FLOW):
        ts = flow_start + i * 0.001
        pkt = (
            Loopback(type=2)
            / IP(src=src_ip, dst=dst_ip)
            / UDP(sport=src_port, dport=dst_port)
            / Raw(load=bytes(rng.getrandbits(8) for _ in range(payload_size)))
        )
        pkt.time = ts
        packets.append(pkt)

    return packets


def main() -> None:
    rng = random.Random(SEED)
    base_ts = 1700000000.0  # Fixed epoch for reproducibility

    all_packets: list = []

    print(f"Generating {NUM_TCP_FLOWS} TCP flows "
          f"({TCP_DATA_PACKETS_PER_FLOW} data + 3 teardown pkts each)...")
    for i in range(NUM_TCP_FLOWS):
        all_packets.extend(_make_tcp_flow(i, rng, base_ts))

    print(f"Generating {NUM_UDP_FLOWS} UDP flows "
          f"({UDP_PACKETS_PER_FLOW} pkts each)...")
    for i in range(NUM_UDP_FLOWS):
        all_packets.extend(_make_udp_flow(i, rng, base_ts))

    # Sort all packets by timestamp for realistic interleaved replay
    all_packets.sort(key=lambda p: float(p.time))

    print(f"Total packets: {len(all_packets)}")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Writing to {os.path.abspath(OUTPUT_PATH)} ...")
    wrpcap(OUTPUT_PATH, all_packets)

    file_size = os.path.getsize(OUTPUT_PATH)
    print(f"Done. File size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
