from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np
import torch
from catboost import CatBoostRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, normalize

from .config import AMRLConfig, Split
from .models import ensemble_predict, train_refinement_ensemble, train_residual_ensemble
from .utils import EPS, centered_refinement, metrics, to_days, validate_split


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class AMRL:
    def __init__(self, config: AMRLConfig = AMRLConfig()):
        self.config = config
        self.device = torch.device(config.device)
        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; use device='cpu'.")
        torch.set_num_threads(config.threads)

    def fit(self, train: Split, valid: Split) -> dict[str, dict[str, float]]:
        validate_split(train, labels=True)
        validate_split(valid, labels=True)
        if tuple(train.visual) != tuple(valid.visual):
            raise ValueError("train and valid must contain the same visual encoders")
        if self.config.temporal:
            if train.time is None or valid.time is None:
                raise ValueError("temporal=True requires posting times")
            if np.any(np.diff(to_days(train.time)) < 0):
                raise ValueError("chronological training data must be sorted by time")

        y_train = np.asarray(train.y, dtype=np.float64)
        y_valid = np.asarray(valid.y, dtype=np.float64)
        self.anchor_model_, anchor_valid, anchor_oof = self._fit_anchor(
            np.asarray(train.anchor), y_train, np.asarray(valid.anchor), y_valid, train.time
        )
        residual_rows = np.flatnonzero(np.isfinite(anchor_oof))
        residual = np.log(np.maximum(y_train[residual_rows], EPS) /
                          np.maximum(anchor_oof[residual_rows], EPS))

        self.visual_pcas_ = {}
        train_visual, valid_visual = {}, {}
        for name in train.visual:
            train_matrix = np.asarray(train.visual[name], dtype=np.float64)
            valid_matrix = np.asarray(valid.visual[name], dtype=np.float64)
            n_components = min(self.config.visual_dim, train_matrix.shape[1], len(train_matrix) - 1)
            solver = "arpack" if n_components < min(train_matrix.shape) else "full"
            pca = PCA(
                n_components=n_components,
                svd_solver=solver,
                random_state=self.config.random_seed,
            )
            self.visual_pcas_[name] = pca
            train_visual[name] = normalize(pca.fit_transform(train_matrix)).astype(np.float32)
            valid_visual[name] = normalize(pca.transform(valid_matrix)).astype(np.float32)

        self.visual_reference_ = {
            name: matrix[residual_rows] for name, matrix in train_visual.items()
        }
        self.visual_residual_reference_ = residual
        self.visual_time_reference_ = (
            to_days(train.time)[residual_rows] if self.config.temporal else None
        )
        train_neighbor = self._neighbor_features(
            self.visual_reference_, self.visual_reference_, residual,
            self.visual_time_reference_, self.visual_time_reference_, same_reference=True
        )
        valid_neighbor = self._neighbor_features(
            self.visual_reference_, valid_visual, residual,
            self.visual_time_reference_,
            to_days(valid.time) if self.config.temporal else None,
            same_reference=False,
        )
        self.visual_model_ = CatBoostRegressor(
            iterations=700, learning_rate=0.03, depth=4, l2_leaf_reg=25,
            loss_function="MAE", random_seed=self.config.random_seed,
            verbose=False, allow_writing_files=False, thread_count=self.config.threads,
        )
        visual_train_x = np.column_stack((train_neighbor, np.asarray(train.anchor)[residual_rows]))
        visual_valid_x = np.column_stack((valid_neighbor, np.asarray(valid.anchor)))
        self.visual_model_.fit(visual_train_x, residual)
        visual_valid = self._bound(self.visual_model_.predict(visual_valid_x))

        self.lexical_stats_, self.lexical_default_ = self._fit_lexical(
            [train.lexical[index] for index in residual_rows], residual
        )
        lexical_valid = self._bound(self._predict_lexical(valid.lexical))

        self.neural_scaler_ = StandardScaler().fit(np.asarray(train.neural)[residual_rows])
        neural_train = self.neural_scaler_.transform(
            np.asarray(train.neural)[residual_rows]
        ).astype(np.float32)
        neural_valid_x = self.neural_scaler_.transform(np.asarray(valid.neural)).astype(np.float32)
        self.residual_models_ = train_residual_ensemble(neural_train, residual, self.config, self.device)
        neural_valid = self._bound(ensemble_predict(self.residual_models_, neural_valid_x, self.device))

        stage1_valid = self._fuse(anchor_valid, visual_valid, lexical_valid, neural_valid)
        refinement_train = np.asarray(
            train.neural if train.refinement is None else train.refinement
        )
        refinement_valid = np.asarray(
            valid.neural if valid.refinement is None else valid.refinement
        )
        self.refinement_scaler_ = StandardScaler().fit(refinement_train)
        refinement_train = self.refinement_scaler_.transform(refinement_train).astype(np.float32)
        refinement_valid = self.refinement_scaler_.transform(refinement_valid).astype(np.float32)
        self.refinement_models_ = train_refinement_ensemble(
            refinement_train, y_train, refinement_valid, y_valid,
            self.config, self.device
        )
        direct_valid = np.exp(np.clip(
            ensemble_predict(self.refinement_models_, refinement_valid, self.device), -20.0, 20.0
        ))

        alpha_scores = {
            alpha: metrics(
                y_valid,
                centered_refinement(stage1_valid, direct_valid, alpha, self.config.output_floor),
            )["MAPE"]
            for alpha in self.config.alpha_candidates
        }
        self.alpha_ = min(alpha_scores, key=alpha_scores.get)
        final_valid = centered_refinement(
            stage1_valid, direct_valid, self.alpha_, self.config.output_floor
        )
        self.validation_components_ = {
            "anchor": anchor_valid,
            "visual": visual_valid,
            "lexical": lexical_valid,
            "neural": neural_valid,
            "stage1": stage1_valid,
            "direct": direct_valid,
            "final": final_valid,
        }
        self.validation_metrics_ = {
            name: metrics(y_valid, prediction)
            for name, prediction in self.validation_components_.items()
            if name in {"anchor", "stage1", "direct", "final"}
        }
        return self.validation_metrics_

    def predict(self, split: Split) -> np.ndarray:
        return self.predict_components(split)["final"]

    def predict_components(self, split: Split) -> dict[str, np.ndarray]:
        validate_split(split, labels=False)
        if tuple(split.visual) != tuple(self.visual_pcas_):
            raise ValueError("prediction split must contain the fitted visual encoders")
        anchor = np.maximum(self.anchor_model_.predict(np.asarray(split.anchor)), EPS)
        visual = {
            name: normalize(self.visual_pcas_[name].transform(np.asarray(matrix))).astype(np.float32)
            for name, matrix in split.visual.items()
        }
        neighbor = self._neighbor_features(
            self.visual_reference_, visual, self.visual_residual_reference_,
            self.visual_time_reference_,
            to_days(split.time) if self.config.temporal else None,
            same_reference=False,
        )
        visual_residual = self._bound(self.visual_model_.predict(
            np.column_stack((neighbor, np.asarray(split.anchor)))
        ))
        lexical_residual = self._bound(self._predict_lexical(split.lexical))
        neural_x = self.neural_scaler_.transform(np.asarray(split.neural)).astype(np.float32)
        neural_residual = self._bound(ensemble_predict(self.residual_models_, neural_x, self.device))
        stage1 = self._fuse(anchor, visual_residual, lexical_residual, neural_residual)
        refinement = np.asarray(split.neural if split.refinement is None else split.refinement)
        refinement = self.refinement_scaler_.transform(refinement).astype(np.float32)
        direct = np.exp(np.clip(
            ensemble_predict(self.refinement_models_, refinement, self.device), -20.0, 20.0
        ))
        final = centered_refinement(stage1, direct, self.alpha_, self.config.output_floor)
        return {
            "anchor": anchor,
            "visual": visual_residual,
            "lexical": lexical_residual,
            "neural": neural_residual,
            "stage1": stage1,
            "direct": direct,
            "final": final,
        }

    def _fit_anchor(self, x_train, y_train, x_valid, y_valid, train_time):
        candidates = []
        for depth in (4, 6):
            for l2 in (10, 30, 100):
                for loss in ("MAE", "RMSE"):
                    model = self._anchor_model(depth, l2, loss, self.config.random_seed)
                    model.fit(x_train, y_train)
                    prediction = np.maximum(model.predict(x_valid), EPS)
                    candidates.append((metrics(y_valid, prediction)["MAPE"], depth, l2, loss, model))
        _, depth, l2, loss, best_model = min(candidates, key=lambda item: item[0])

        oof = np.full(len(y_train), np.nan)
        splitter = (
            TimeSeriesSplit(n_splits=self.config.oof_splits)
            if self.config.temporal
            else KFold(self.config.oof_splits, shuffle=True, random_state=self.config.random_seed)
        )
        times = to_days(train_time) if self.config.temporal else None
        for fold, (fit_rows, holdout_rows) in enumerate(splitter.split(x_train)):
            if self.config.temporal:
                fit_rows = fit_rows[times[fit_rows] < times[holdout_rows].min()]
            model = self._anchor_model(depth, l2, loss, self.config.random_seed + fold + 1)
            model.fit(x_train[fit_rows], y_train[fit_rows])
            oof[holdout_rows] = np.maximum(model.predict(x_train[holdout_rows]), EPS)
        return best_model, np.maximum(best_model.predict(x_valid), EPS), oof

    def _anchor_model(self, depth, l2, loss, seed):
        return CatBoostRegressor(
            iterations=700, learning_rate=0.035, depth=depth,
            l2_leaf_reg=l2, loss_function=loss, random_seed=seed,
            verbose=False, allow_writing_files=False, thread_count=self.config.threads,
        )

    def _neighbor_features(
        self, reference, query, residual, reference_time, query_time, same_reference
    ):
        columns = []
        fallback = float(np.mean(residual))
        for name in reference:
            ref, current = reference[name], query[name]
            result = np.full((len(current), len(self.config.neighbors)), fallback, dtype=np.float32)
            for row, vector in enumerate(current):
                similarity = ref @ vector
                eligible = np.ones(len(ref), dtype=bool)
                if self.config.temporal:
                    age = query_time[row] - reference_time
                    eligible &= age > 0
                elif same_reference:
                    eligible[row] = False
                candidates = np.flatnonzero(eligible)
                if not len(candidates):
                    continue
                ranked = candidates[np.argsort(similarity[candidates])[::-1]]
                for column, k in enumerate(self.config.neighbors):
                    selected = ranked[: min(k, len(ranked))]
                    weight = (np.maximum(similarity[selected], 0.0) + EPS) ** self.config.similarity_power
                    if self.config.temporal:
                        weight *= np.exp(-(query_time[row] - reference_time[selected]) /
                                         self.config.time_decay_days)
                    result[row, column] = np.average(residual[selected], weights=weight)
            columns.append(result)
        return np.column_stack(columns)

    def _fit_lexical(self, records, residual):
        document_count = Counter()
        weighted_count = defaultdict(float)
        weighted_total = defaultdict(float)
        default = float(np.mean(residual))
        for record, target in zip(records, residual):
            token_weights = self._tokens(record)
            document_count.update(token_weights)
            for token, weight in token_weights.items():
                weighted_count[token] += weight
                weighted_total[token] += weight * float(target)
        smoothing = self.config.lexical_smoothing
        stats = {
            token: (weighted_total[token] + smoothing * default) /
                   (weighted_count[token] + smoothing)
            for token in document_count
            if document_count[token] >= self.config.lexical_min_count
        }
        return stats, default

    def _predict_lexical(self, records):
        output = []
        for record in records:
            token_weights = self._tokens(record)
            available = [(self.lexical_stats_[token], weight)
                         for token, weight in token_weights.items()
                         if token in self.lexical_stats_]
            output.append(
                np.average([value for value, _ in available],
                           weights=[weight for _, weight in available])
                if available else self.lexical_default_
            )
        return np.asarray(output)

    @staticmethod
    def _tokens(record):
        source_weights = {
            "content": 2.0, "tags": 1.5, "music": 1.0,
            "asr": 1.0, "ocr": 1.0, "caption": 1.0,
        }
        output = Counter()
        for source, source_weight in source_weights.items():
            value = record.get(source, "")
            if isinstance(value, (list, tuple, np.ndarray)):
                value = " ".join(map(str, value))
            for token in TOKEN_RE.findall(str(value).lower())[:100]:
                if len(token) > 1 and not token.isdigit():
                    output[token] += source_weight
        return output

    def _fuse(self, anchor, visual, lexical, neural):
        visual_weight, lexical_weight, neural_weight = self.config.residual_weights
        output = np.zeros_like(anchor, dtype=np.float64)
        for mix, omega in zip(self.config.lexical_mix, self.config.lexical_omegas):
            correction = (visual_weight * visual + lexical_weight * omega * lexical +
                          neural_weight * neural)
            output += mix * anchor * np.exp(correction)
        return output

    def _bound(self, residual):
        return np.clip(np.asarray(residual), -self.config.residual_bound,
                       self.config.residual_bound)


def _self_check() -> None:
    stage1 = np.array([2.0, 5.0, 11.0])
    direct = np.array([3.0, 4.0, 15.0])
    refined = centered_refinement(stage1, direct, alpha=0.03, output_floor=0.0)
    assert np.isclose(np.log(refined).mean(), np.log(stage1).mean())
    print("AMRL self-check passed")


if __name__ == "__main__":
    _self_check()
