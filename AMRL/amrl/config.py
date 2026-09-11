from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass
class Split:
    """Pre-extracted features for one dataset split."""

    anchor: np.ndarray
    visual: Mapping[str, np.ndarray]
    neural: np.ndarray
    lexical: Sequence[Mapping[str, object]]
    y: np.ndarray | None = None
    time: np.ndarray | None = None
    refinement: np.ndarray | None = None


@dataclass(frozen=True)
class AMRLConfig:
    """AMRL defaults used in the paper experiments."""

    temporal: bool = False
    oof_splits: int = 5
    random_seed: int = 2026
    threads: int = 8
    visual_dim: int = 128
    neighbors: tuple[int, ...] = (3, 10, 25, 50)
    similarity_power: float = 2.0
    time_decay_days: float = 120.0
    residual_bound: float = 0.8
    lexical_smoothing: float = 5.0
    lexical_min_count: int = 2
    residual_weights: tuple[float, float, float] = (0.0471, 0.05, 0.02)
    lexical_omegas: tuple[float, float] = (0.975, 0.950)
    lexical_mix: tuple[float, float] = (0.8, 0.2)
    residual_seeds: tuple[int, ...] = (2126, 2127, 2128)
    refinement_seeds: tuple[int, ...] = (42, 43, 44, 45, 46)
    alpha_candidates: tuple[float, ...] = (0.01, 0.02, 0.03, 0.05, 0.08)
    output_floor: float = 1.0
    device: str = "cuda"
