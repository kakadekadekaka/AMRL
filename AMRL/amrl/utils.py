from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from .config import Split


EPS = 1e-8


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    prediction = np.maximum(np.asarray(prediction, dtype=np.float64), EPS)
    return {
        "MAPE": float(np.mean(np.abs(y - prediction) / np.maximum(np.abs(y), EPS))),
        "SRC": float(spearmanr(y, prediction).statistic),
    }


def centered_refinement(
    stage1: np.ndarray,
    direct: np.ndarray,
    alpha: float,
    output_floor: float,
) -> np.ndarray:
    """Apply a centered log-ratio correction to a prediction batch."""
    stage1 = np.maximum(np.asarray(stage1, dtype=np.float64), EPS)
    direct = np.maximum(np.asarray(direct, dtype=np.float64), EPS)
    correction = np.log(direct) - np.log(stage1)
    correction -= correction.mean()
    return np.maximum(stage1 * np.exp(alpha * correction), output_floor)


def to_days(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[s]").astype(np.int64) / 86400.0
    return values.astype(np.float64)


def validate_split(split: Split, labels: bool) -> None:
    size = len(split.anchor)
    arrays = [split.neural, *split.visual.values()]
    if split.refinement is not None:
        arrays.append(split.refinement)
    if any(len(value) != size for value in arrays) or len(split.lexical) != size:
        raise ValueError("all features in a split must contain the same number of rows")
    if labels and (split.y is None or len(split.y) != size):
        raise ValueError("training and validation splits require one label per row")
    if split.y is not None and np.any(np.asarray(split.y) <= 0):
        raise ValueError("AMRL requires positive popularity labels")
