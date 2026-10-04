"""
Golden Holdout Dataset Generator — ChronoGuard M2

Generates a synthetic golden holdout CSV for the accuracy gate mechanism.
This is scaffolding — will be replaced by a real holdout derived from
the CIC-IDS2018/2019 dataset.

Composition:
    - 200 samples total (100 benign, 100 attack) — balanced binary.
    - Features match FEATURE_ORDER in config.py exactly.
    - Random seed 42 for reproducibility.
    - Feature distributions are realistic ranges based on typical network
      flow characteristics (not random noise).

Usage:
    python scripts/generate_golden_holdout.py

Output:
    golden_holdout.csv (in the project root)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# Add project root to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import FEATURE_ORDER

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "golden_holdout.csv")
NUM_SAMPLES = 200
SEED = 42


def main() -> None:
    print(f"[Golden Holdout] Generating synthetic holdout dataset...")
    print(f"  Samples     : {NUM_SAMPLES} (balanced binary)")
    print(f"  Features    : {len(FEATURE_ORDER)}")
    print(f"  Seed        : {SEED}")
    print(f"  Output      : {OUTPUT_PATH}")

    rng = np.random.RandomState(SEED)

    # Generate realistic feature distributions
    n_benign = NUM_SAMPLES // 2
    n_attack = NUM_SAMPLES - n_benign

    data = {}

    # Benign flows: moderate duration, moderate packet rates
    # Attack flows: either very short (scans) or very long (exfil), extreme rates
    base_ts = 1700000000.0

    # flow_start_ts
    benign_start = base_ts + rng.uniform(0, 3600, n_benign)
    attack_start = base_ts + rng.uniform(0, 3600, n_attack)
    data["flow_start_ts"] = np.concatenate([benign_start, attack_start])

    # flow_duration (benign: 0.5-30s; attack: 0.01-2s or 60-300s)
    benign_dur = rng.uniform(0.5, 30.0, n_benign)
    attack_dur = np.where(
        rng.random(n_attack) < 0.5,
        rng.uniform(0.01, 2.0, n_attack),
        rng.uniform(60.0, 300.0, n_attack),
    )
    data["flow_duration"] = np.concatenate([benign_dur, attack_dur])

    # flow_end_ts = start + duration
    data["flow_end_ts"] = data["flow_start_ts"] + data["flow_duration"]

    # flow_iat_mean (benign: 0.1-1.0s; attack: 0.001-0.01s or 5-10s)
    data["flow_iat_mean"] = np.concatenate([
        rng.uniform(0.1, 1.0, n_benign),
        np.where(
            rng.random(n_attack) < 0.5,
            rng.uniform(0.001, 0.01, n_attack),
            rng.uniform(5.0, 10.0, n_attack),
        ),
    ])

    # flow_iat_std
    data["flow_iat_std"] = np.concatenate([
        rng.uniform(0.01, 0.5, n_benign),
        rng.uniform(0.0, 0.1, n_attack),
    ])

    # fwd_packet_count
    data["fwd_packet_count"] = np.concatenate([
        rng.randint(5, 50, n_benign),
        rng.randint(1, 1000, n_attack),
    ]).astype(float)

    # bwd_packet_count
    data["bwd_packet_count"] = np.concatenate([
        rng.randint(3, 40, n_benign),
        rng.randint(0, 500, n_attack),
    ]).astype(float)

    # fwd_packet_length_max
    data["fwd_packet_length_max"] = np.concatenate([
        rng.randint(100, 1500, n_benign),
        rng.randint(40, 1500, n_attack),
    ]).astype(float)

    # fwd_packet_length_mean
    data["fwd_packet_length_mean"] = np.concatenate([
        rng.uniform(100, 800, n_benign),
        rng.uniform(40, 1200, n_attack),
    ])

    # bwd_packet_length_max
    data["bwd_packet_length_max"] = np.concatenate([
        rng.randint(100, 1500, n_benign),
        rng.randint(0, 1500, n_attack),
    ]).astype(float)

    # bwd_packet_length_mean
    data["bwd_packet_length_mean"] = np.concatenate([
        rng.uniform(50, 600, n_benign),
        rng.uniform(0, 800, n_attack),
    ])

    # fwd_packets_per_sec
    safe_dur = np.maximum(data["flow_duration"], 0.001)
    data["fwd_packets_per_sec"] = data["fwd_packet_count"] / safe_dur

    # bwd_packets_per_sec
    data["bwd_packets_per_sec"] = data["bwd_packet_count"] / safe_dur

    # protocol (6=TCP mostly, some 17=UDP)
    data["protocol"] = np.concatenate([
        np.where(rng.random(n_benign) < 0.8, 6, 17),
        np.where(rng.random(n_attack) < 0.6, 6, 17),
    ]).astype(float)

    # Labels: 0=benign, 1=attack
    labels = np.concatenate([np.zeros(n_benign), np.ones(n_attack)]).astype(int)

    # Build DataFrame
    df = pd.DataFrame(data)

    # Ensure columns are in FEATURE_ORDER
    df = df[FEATURE_ORDER]
    df["label"] = labels

    # Shuffle
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"[Golden Holdout] Generated {len(df)} samples → {OUTPUT_PATH}")
    print(f"  Class balance: {(labels == 0).sum()} benign, {(labels == 1).sum()} attack")
    print(f"  Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
