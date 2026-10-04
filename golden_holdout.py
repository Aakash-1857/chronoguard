"""
ChronoGuard v1.0 — Golden Holdout Evaluator (M2)

Loads a fixed, versioned, read-only golden holdout dataset and evaluates
a candidate model's accuracy against it.

Spec references: M2 §5.3 (evaluate signature), §5.5 (holdout properties).

The golden holdout set is NEVER written to by this module — read-only
access pattern enforced throughout.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import FEATURE_ORDER

logger = logging.getLogger(__name__)


def evaluate(model, holdout_path: str) -> float:
    """Evaluate a LightGBM Booster against the protected golden holdout set.

    Parameters
    ----------
    model : lightgbm.Booster
        The candidate model to evaluate.
    holdout_path : str
        Path to the golden holdout CSV.  Must contain columns matching
        FEATURE_ORDER plus a 'label' column (0=benign, 1=attack).

    Returns
    -------
    float
        Accuracy (0.0–1.0) of the model on the holdout set.

    Raises
    ------
    FileNotFoundError
        If holdout_path does not exist.
    ValueError
        If required columns are missing from the holdout CSV.
    """
    # Read-only access — never write to holdout_path
    df = pd.read_csv(holdout_path)

    # Validate required columns
    missing_features = [f for f in FEATURE_ORDER if f not in df.columns]
    if missing_features:
        raise ValueError(
            f"Golden holdout missing required features: {missing_features}"
        )
    if "label" not in df.columns:
        raise ValueError("Golden holdout missing 'label' column")

    X = df[FEATURE_ORDER].values.astype(np.float32)
    y_true = df["label"].values.astype(int)

    # Predict using the LightGBM Booster
    raw_preds = model.predict(X)

    # For binary classification, LightGBM returns probabilities;
    # threshold at 0.5 for class labels.
    if raw_preds.ndim == 1:
        y_pred = (raw_preds >= 0.5).astype(int)
    else:
        # Multi-class: argmax
        y_pred = np.argmax(raw_preds, axis=1).astype(int)

    accuracy = float(np.mean(y_pred == y_true))

    logger.info(
        "Golden holdout evaluation: accuracy=%.4f (%d/%d correct)",
        accuracy,
        int(np.sum(y_pred == y_true)),
        len(y_true),
    )

    return accuracy
