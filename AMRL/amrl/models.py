from __future__ import annotations

import copy
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import AMRLConfig
from .utils import EPS, metrics


class ResidualMLP(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(0.25),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.SiLU(), nn.Dropout(0.25),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class RefinementMLP(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.30),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.30),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.30),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def predict_mlp(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).to(device)
            output.append(model(batch).cpu().numpy())
    return np.concatenate(output)


def train_residual_ensemble(
    features: np.ndarray,
    targets: np.ndarray,
    config: AMRLConfig,
    device: torch.device,
) -> list[ResidualMLP]:
    dataset = TensorDataset(
        torch.from_numpy(features), torch.from_numpy(targets.astype(np.float32))
    )
    models = []
    for seed in config.residual_seeds:
        seed_all(seed)
        model = ResidualMLP(features.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-3)
        loader = DataLoader(dataset, batch_size=256, shuffle=True)
        for _ in range(80):
            model.train()
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.smooth_l1_loss(model(x), y, beta=0.08)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                optimizer.step()
        models.append(copy.deepcopy(model).cpu())
    return models


def train_refinement_ensemble(
    features: np.ndarray,
    targets: np.ndarray,
    valid_features: np.ndarray,
    valid_targets: np.ndarray,
    config: AMRLConfig,
    device: torch.device,
) -> list[RefinementMLP]:
    dataset = TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(np.log(np.maximum(targets, EPS)).astype(np.float32)),
    )
    models = []
    for seed in config.refinement_seeds:
        seed_all(seed)
        model = RefinementMLP(features.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        loader = DataLoader(dataset, batch_size=128, shuffle=True)
        best_score, best_state, patience = float("inf"), None, 0
        for _ in range(200):
            model.train()
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.huber_loss(model(x), y, delta=1.0)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            log_prediction = np.clip(predict_mlp(model, valid_features, device), -20.0, 20.0)
            score = metrics(valid_targets, np.exp(log_prediction))["MAPE"]
            if score < best_score:
                best_score, best_state, patience = score, copy.deepcopy(model.state_dict()), 0
            else:
                patience += 1
            if patience >= 30:
                break
        model.load_state_dict(best_state)
        models.append(copy.deepcopy(model).cpu())
    return models


def ensemble_predict(
    models: Sequence[nn.Module], features: np.ndarray, device: torch.device
) -> np.ndarray:
    predictions = []
    for model in models:
        model = model.to(device)
        predictions.append(predict_mlp(model, features, device))
        model.cpu()
    return np.mean(predictions, axis=0)
