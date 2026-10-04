"""
ChronoGuard v1.0 — Sniffer Thread (M1)

Captures live packets off the configured interface in a dedicated thread,
aggregates them into stateful 5-tuple flow records via FlowTable, and
pushes finalized FlowRecords onto a thread-safe queue for the engine.

The sniffer also runs periodic eviction sweeps (every ~EVICTION_INTERVAL
seconds) to finalize flows that have exceeded the inactivity timeout,
even when no new packets arrive.

This module does NOT launch tcpreplay — that is operator-invoked externally
(spec §3).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import TYPE_CHECKING

from scapy.all import sniff as scapy_sniff
from scapy.layers.inet import IP

from config import EVICTION_INTERVAL, INTERFACE
from flow_state import FlowRecord, FlowTable

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def run_sniffer(
    interface: str,
    out_queue: queue.Queue[FlowRecord],
    stop_event: threading.Event,
) -> None:
    """Packet capture loop — runs in a dedicated thread.

    Signature matches spec §5.5.

    Parameters
    ----------
    interface : str
        Network interface to capture on (e.g., ``lo0`` on macOS, ``lo`` on Linux).
    out_queue : queue.Queue[FlowRecord]
        Thread-safe queue to push finalized flow records onto.
    stop_event : threading.Event
        Cooperative shutdown signal shared with the engine thread and main.
    """
    flow_table = FlowTable()
    last_eviction_ts: float = time.monotonic()
    packet_count: int = 0

    def _enqueue_record(record: FlowRecord) -> None:
        """Push a finalized FlowRecord onto the output queue."""
        try:
            out_queue.put_nowait(record)
        except queue.Full:
            logger.warning(
                "Output queue full — dropping flow %s", record["flow_id"]
            )

    def _process_packet(pkt) -> None:  # noqa: ANN001 — scapy packet type
        """Callback invoked by scapy for each captured packet."""
        nonlocal last_eviction_ts, packet_count

        packet_count += 1

        # Ingest packet into the flow table; may return a finalized record
        # on FIN/RST (spec §5.5).
        try:
            record = flow_table.ingest_packet(pkt)
            if record is not None:
                _enqueue_record(record)
        except Exception:
            # Spec §6: malformed/non-IP packets — skip silently at debug level.
            logger.debug("Error processing packet, skipping", exc_info=True)

        # Periodic eviction sweep (spec §6: every 1–2 seconds).
        now = time.monotonic()
        if (now - last_eviction_ts) >= EVICTION_INTERVAL:
            last_eviction_ts = now
            # Use wall-clock time for eviction comparison (packet timestamps
            # are wall-clock epoch seconds from scapy).
            wall_now = time.time()
            try:
                stale_records = flow_table.evict_stale(wall_now)
                for rec in stale_records:
                    _enqueue_record(rec)
            except Exception:
                logger.debug("Error during eviction sweep", exc_info=True)

    # --- Startup ---
    logger.info("Sniffer starting on interface: %s", interface)

    try:
        # scapy.sniff with stop_filter callback for cooperative shutdown.
        # `store=False` avoids accumulating packets in memory.
        # `prn` callback processes each packet inline.
        scapy_sniff(
            iface=interface,
            prn=_process_packet,
            store=False,
            stop_filter=lambda _pkt: stop_event.is_set(),
        )
    except PermissionError:
        # Spec §6: sniffer interface bind failure at startup — fatal.
        logger.error(
            "FATAL: Permission denied binding to interface '%s'. "
            "Raw socket capture typically requires sudo or CAP_NET_RAW.",
            interface,
        )
        stop_event.set()
        return
    except OSError as exc:
        # Spec §6: sniffer interface bind failure at startup — fatal.
        logger.error(
            "FATAL: Cannot bind to interface '%s': %s",
            interface,
            exc,
        )
        stop_event.set()
        return
    except Exception:
        # Spec §6: uncaught exceptions must be logged, and stop_event set
        # so other threads can shut down cleanly.
        logger.error(
            "Sniffer thread encountered an unhandled exception",
            exc_info=True,
        )
        stop_event.set()
        return

    # --- Shutdown: final eviction sweep ---
    # Finalize any remaining flows as TIMEOUT on shutdown.
    wall_now = time.time()
    remaining = flow_table.evict_stale(wall_now)
    # Also drain any flows that haven't timed out yet (they won't be
    # finalized by evict_stale since they're still within the timeout window).
    # We force-finalize them with TIMEOUT since the capture is ending.
    # To do this, we temporarily set timeout to 0 and sweep again.
    flow_table._inactivity_timeout = 0.0
    leftover = flow_table.evict_stale(wall_now)
    for rec in remaining + leftover:
        _enqueue_record(rec)

    logger.info(
        "Sniffer stopped. Total packets captured: %d, "
        "remaining flows finalized: %d",
        packet_count,
        len(remaining) + len(leftover),
    )
