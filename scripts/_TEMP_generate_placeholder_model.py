"""
=============================================================================
TEMPORARY SCAFFOLDING — NOT PRODUCTION CODE
=============================================================================

This script generates a trivial placeholder ONNX model for M1 pipeline
testing ONLY.  It produces a syntactically valid ONNX artifact with the
correct input shape (1, 14) matching FEATURE_ORDER in config.py.

The model is a RandomForestClassifier trained on random noise — it has
NO real predictive value.  It will be replaced by a properly trained
model artifact before M2.

Dependencies (NOT required at runtime):
    pip install scikit-learn skl2onnx

Usage:
    python scripts/_TEMP_generate_placeholder_model.py

Output:
    model.onnx  (in the project root)
=============================================================================
"""

from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier

# skl2onnx for ONNX conversion
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# ---------------------------------------------------------------------------
# Configuration — must match config.py FEATURE_ORDER
# ---------------------------------------------------------------------------
NUM_FEATURES = 14
NUM_SAMPLES = 500
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model.onnx")


def main() -> None:
    print(f"[TEMP SCAFFOLDING] Generating placeholder ONNX model...")
    print(f"  Features : {NUM_FEATURES}")
    print(f"  Samples  : {NUM_SAMPLES}")
    print(f"  Output   : {OUTPUT_PATH}")

    # Generate random training data
    rng = np.random.RandomState(42)
    X = rng.randn(NUM_SAMPLES, NUM_FEATURES).astype(np.float32)
    # Binary classification: 0 = benign, 1 = attack
    y = rng.randint(0, 2, size=NUM_SAMPLES)

    # Train a trivial model
    clf = RandomForestClassifier(
        n_estimators=5,
        max_depth=3,
        random_state=42,
    )
    clf.fit(X, y)

    # Convert to ONNX
    initial_type = [("input", FloatTensorType([None, NUM_FEATURES]))]
    onnx_model = convert_sklearn(
        clf,
        initial_types=initial_type,
        target_opset=13,
    )

    # Save
    with open(OUTPUT_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"[TEMP SCAFFOLDING] Placeholder model saved to: {OUTPUT_PATH}")
    print(f"[TEMP SCAFFOLDING] This model is NOT suitable for production use.")


if __name__ == "__main__":
    main()
