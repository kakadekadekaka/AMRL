"""Prepare labeled SMP data for the AMRL implementation.

The original competition test set is not used. The 4,000 labeled training
posts are sorted by publication time and split chronologically into 80/10/10.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


SPLITS = ("train", "valid", "test")
CORE_TABLES = ("posts", "labels", "users", "videos")
CATEGORICAL_COLUMNS = (
    "post_location",
    "post_text_language",
    "asr_language",
    "video_ratio",
    "video_format",
)
LEXICAL_SOURCES = ("content", "tags", "music", "asr", "ocr", "caption")
AUXILIARY_FILES = {
    "video_asr.csv": ("asr_text", "asr_language", "asr_char_count"),
    "frame_ocr_rapid.csv": ("ocr_text", "ocr_raw_text_count"),
    "blip_video_captions_f01234567_t24.csv": tuple(
        f"blip_caption_{index}" for index in range(8)
    ),
    "video_stats.csv": (
        "duration", "fps", "width", "height", "aspect_ratio", "total_frames"
    ),
    "video_file_props.csv": ("file_size_bytes", "file_exists"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, np.ndarray)):
        return " ".join(map(str, value))
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _read_labeled_posts(data_dir: Path) -> pd.DataFrame:
    paths = {name: data_dir / f"{name}_train.parquet" for name in CORE_TABLES}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing SMP files: " + ", ".join(missing))

    tables = {name: pd.read_parquet(path) for name, path in paths.items()}
    for name, key in (
        ("posts", "pid"), ("labels", "pid"), ("users", "uid"), ("videos", "pid")
    ):
        if tables[name][key].isna().any() or not tables[name][key].is_unique:
            raise ValueError(f"{name}: {key} must be non-null and unique")

    posts = tables["posts"].copy()
    posts["post_time"] = pd.to_datetime(posts["post_time"], errors="raise")
    if posts[["uid", "post_time"]].isna().any().any():
        raise ValueError("posts: uid and post_time must be non-null")
    if set(posts["pid"]) != set(tables["labels"]["pid"]):
        raise ValueError("labels: pid set differs from posts")
    if set(posts["pid"]) != set(tables["videos"]["pid"]):
        raise ValueError("videos: pid set differs from posts")
    if not set(posts["uid"]).issubset(set(tables["users"]["uid"])):
        raise ValueError("users: missing users referenced by posts")

    frame = posts.merge(
        tables["users"], on="uid", how="left", validate="many_to_one", indicator="_user"
    )
    if not frame["_user"].eq("both").all():
        raise ValueError("users failed to align with posts")
    frame = frame.drop(columns="_user").merge(
        tables["videos"],
        on=["pid", "uid"],
        how="left",
        validate="one_to_one",
        indicator="_video",
    )
    if not frame["_video"].eq("both").all():
        raise ValueError("videos failed to align with posts")
    frame = frame.drop(columns="_video").merge(
        tables["labels"][["pid", "uid", "popularity"]],
        on=["pid", "uid"],
        how="left",
        validate="one_to_one",
        indicator="_label",
    )
    if not frame["_label"].eq("both").all():
        raise ValueError("labels failed to align with posts")
    frame = frame.drop(columns="_label")
    if (frame["popularity"] <= 0).any():
        raise ValueError("AMRL requires positive popularity labels")
    return frame.sort_values(["post_time", "pid"]).reset_index(drop=True)


def chronological_split(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return an exact 80/10/10 split after chronological sorting."""
    ordered = frame.sort_values(["post_time", "pid"]).reset_index(drop=True)
    train_end = len(ordered) * 8 // 10
    valid_end = len(ordered) * 9 // 10
    if not 0 < train_end < valid_end < len(ordered):
        raise ValueError("At least ten samples are required for an 80/10/10 split")
    return {
        "train": ordered.iloc[:train_end].copy(),
        "valid": ordered.iloc[train_end:valid_end].copy(),
        "test": ordered.iloc[valid_end:].copy(),
    }


def _attach_auxiliary(
    frame: pd.DataFrame, aux_dir: Path | None
) -> tuple[pd.DataFrame, list[Path]]:
    if aux_dir is None:
        return frame, []
    used = []
    for filename, columns in AUXILIARY_FILES.items():
        path = aux_dir / filename
        if not path.is_file():
            continue
        auxiliary = pd.read_csv(path)
        if "pid" not in auxiliary or auxiliary["pid"].duplicated().any():
            raise ValueError(f"{path}: pid must be present and unique")
        selected = [
            column for column in columns if column in auxiliary and column not in frame
        ]
        frame = frame.merge(
            auxiliary[["pid", *selected]], on="pid", how="left", validate="one_to_one"
        )
        used.append(path)
    return frame, used


def _load_visual(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            values = pickle.load(handle)
        if not isinstance(values, dict):
            raise TypeError(f"{path}: expected a pid-to-vector dictionary")
        return {
            str(pid): np.asarray(vector, dtype=np.float32).reshape(-1)
            for pid, vector in values.items()
        }
    if path.suffix.lower() == ".csv":
        table = pd.read_csv(path)
    elif path.suffix.lower() == ".parquet":
        table = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported visual feature file: {path}")
    if "pid" not in table or table["pid"].duplicated().any():
        raise ValueError(f"{path}: pid must be present and unique")
    matrix = table.drop(columns="pid").to_numpy(dtype=np.float32)
    return {str(pid): row for pid, row in zip(table["pid"], matrix)}


def _align_visual(
    features: dict[str, np.ndarray], pids: pd.Series, name: str
) -> tuple[np.ndarray, int]:
    if not features:
        raise ValueError(f"{name}: visual feature file is empty")
    dimensions = {len(vector) for vector in features.values()}
    if len(dimensions) != 1:
        raise ValueError(f"{name}: visual vectors must have one fixed dimension")
    zero = np.zeros(dimensions.pop(), dtype=np.float32)
    missing = sum(str(pid) not in features for pid in pids)
    matrix = np.stack([features.get(str(pid), zero) for pid in pids]).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name}: visual features contain non-finite values")
    return matrix, missing


def _lexical_records(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    def column(name: str) -> list[str]:
        if name not in frame:
            return [""] * len(frame)
        return [_as_text(value) for value in frame[name]]

    caption_columns = [name for name in frame if name.startswith("blip_caption_")]
    if caption_columns:
        captions = [
            " ".join(_as_text(row[name]) for name in caption_columns).strip()
            for _, row in frame.iterrows()
        ]
    else:
        captions = [""] * len(frame)
    return {
        "content": np.asarray(column("post_content"), dtype=str),
        "tags": np.asarray(column("post_suggested_words"), dtype=str),
        "music": np.asarray(column("music_title"), dtype=str),
        "asr": np.asarray(column("asr_text"), dtype=str),
        "ocr": np.asarray(column("ocr_text"), dtype=str),
        "caption": np.asarray(captions, dtype=str),
    }


def _structured_features(
    splits: dict[str, pd.DataFrame], lexical: dict[str, dict[str, np.ndarray]]
) -> tuple[dict[str, np.ndarray], list[str]]:
    train = splits["train"]
    excluded = {"popularity"}
    excluded.update(name for name in train if name.startswith("blip_caption_"))
    numeric_columns = [
        name
        for name in train.select_dtypes(include=["number", "bool"]).columns
        if name not in excluded
    ]
    medians = {}
    for name in numeric_columns:
        values = pd.to_numeric(train[name], errors="coerce")
        medians[name] = float(values.median()) if values.notna().any() else 0.0
    category_values = {
        name: sorted(train[name].fillna("<missing>").astype(str).unique())
        for name in CATEGORICAL_COLUMNS
        if name in train
    }
    start = train["post_time"].min()
    names = [f"log1p_{name}" for name in numeric_columns]
    names += [
        "days_since_start", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"
    ]
    names += [f"log1p_{source}_chars" for source in LEXICAL_SOURCES]
    names += [
        f"{column}={value}"
        for column, values in category_values.items()
        for value in values
    ]

    matrices = {}
    for split, frame in splits.items():
        blocks = []
        if numeric_columns:
            numeric = np.column_stack([
                pd.to_numeric(frame[name], errors="coerce")
                .fillna(medians[name])
                .to_numpy(float)
                for name in numeric_columns
            ])
            blocks.append(np.sign(numeric) * np.log1p(np.abs(numeric)))
        age = (frame["post_time"] - start).dt.total_seconds().to_numpy() / 86400.0
        hour = frame["post_time"].dt.hour.to_numpy()
        weekday = frame["post_time"].dt.dayofweek.to_numpy()
        blocks.append(np.column_stack((
            age,
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * weekday / 7),
            np.cos(2 * np.pi * weekday / 7),
        )))
        blocks.append(np.column_stack([
            np.log1p(np.char.str_len(lexical[split][source]).astype(float))
            for source in LEXICAL_SOURCES
        ]))
        for column, values in category_values.items():
            current = frame[column].fillna("<missing>").astype(str).to_numpy()
            blocks.append(np.column_stack([current == value for value in values]))
        matrices[split] = np.column_stack(blocks).astype(np.float32)
    return matrices, names


def _save_split(
    path: Path,
    frame: pd.DataFrame,
    anchor: np.ndarray,
    neural: np.ndarray,
    visual: dict[str, np.ndarray],
    lexical: dict[str, np.ndarray],
    include_labels: bool,
) -> None:
    payload = {
        "pid": frame["pid"].astype(str).to_numpy(dtype=str),
        "time": frame["post_time"].to_numpy(dtype="datetime64[s]"),
        "anchor": anchor,
        "neural": neural,
    }
    payload.update({f"visual__{name}": matrix for name, matrix in visual.items()})
    payload.update({f"lexical__{name}": values for name, values in lexical.items()})
    if include_labels:
        payload["y"] = frame["popularity"].to_numpy(dtype=np.float64)
    np.savez_compressed(path, **payload)


def prepare_smp(
    data_dir: Path,
    output_dir: Path,
    visual_specs: dict[str, Path] | None = None,
    aux_dir: Path | None = None,
) -> dict:
    frame = _read_labeled_posts(data_dir)
    frame, auxiliary_paths = _attach_auxiliary(frame, aux_dir)
    splits = chronological_split(frame)
    lexical = {name: _lexical_records(part) for name, part in splits.items()}
    anchors, anchor_names = _structured_features(splits, lexical)

    if visual_specs is None:
        visual_specs = {"visual": data_dir / "visual_features.pkl"}
    if not visual_specs:
        raise ValueError("At least one visual feature file is required")
    loaded_visual = {name: _load_visual(path) for name, path in visual_specs.items()}
    aligned_visual = {}
    missing_visual = {}
    for split, part in splits.items():
        aligned_visual[split] = {}
        missing_visual[split] = {}
        for name, features in loaded_visual.items():
            matrix, missing = _align_visual(features, part["pid"], name)
            aligned_visual[split][name] = matrix
            missing_visual[split][name] = missing

    reduced_visual = {split: {} for split in SPLITS}
    reduced_dimensions = {}
    for name in visual_specs:
        train_matrix = aligned_visual["train"][name]
        dimensions = min(128, train_matrix.shape[1], len(train_matrix) - 1)
        solver = "arpack" if dimensions < min(train_matrix.shape) else "full"
        reducer = PCA(
            n_components=dimensions,
            svd_solver=solver,
            random_state=2026,
        )
        reduced_visual["train"][name] = reducer.fit_transform(
            train_matrix.astype(np.float64)
        )
        for split in ("valid", "test"):
            reduced_visual[split][name] = reducer.transform(
                aligned_visual[split][name].astype(np.float64)
            )
        reduced_dimensions[name] = dimensions
    neural = {
        split: np.column_stack([
            anchors[split], *reduced_visual[split].values()
        ]).astype(np.float32)
        for split in SPLITS
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        _save_split(
            output_dir / f"{split}.npz",
            splits[split],
            anchors[split],
            neural[split],
            aligned_visual[split],
            lexical[split],
            include_labels=split != "test",
        )
    np.savez_compressed(
        output_dir / "test_labels.npz",
        pid=splits["test"]["pid"].astype(str).to_numpy(dtype=str),
        y=splits["test"]["popularity"].to_numpy(dtype=np.float64),
    )
    manifest = pd.concat([
        part[["pid", "uid", "post_time"]].assign(split=split)
        for split, part in splits.items()
    ], ignore_index=True)
    manifest.to_csv(output_dir / "split_manifest.csv", index=False)

    core_paths = [data_dir / f"{name}_train.parquet" for name in CORE_TABLES]
    all_sources = [*core_paths, *visual_specs.values(), *auxiliary_paths]
    metadata = {
        "protocol": "chronological 80/10/10 split of the labeled SMP training set",
        "sort_by": ["post_time", "pid"],
        "splits": {
            name: {
                "samples": len(part),
                "time_min": str(part["post_time"].min()),
                "time_max": str(part["post_time"].max()),
            }
            for name, part in splits.items()
        },
        "boundary_time_ties": {
            "train_valid": bool(
                splits["train"]["post_time"].max()
                == splits["valid"]["post_time"].min()
            ),
            "valid_test": bool(
                splits["valid"]["post_time"].max()
                == splits["test"]["post_time"].min()
            ),
        },
        "anchor_features": anchor_names,
        "visual_features": {
            name: {
                "path": str(path.resolve()),
                "dimension": int(aligned_visual["train"][name].shape[1]),
                "neural_pca_dimension": reduced_dimensions[name],
                "missing_by_split": {
                    split: missing_visual[split][name] for split in SPLITS
                },
            }
            for name, path in visual_specs.items()
        },
        "auxiliary_files": [str(path.resolve()) for path in auxiliary_paths],
        "source_sha256": {str(path.resolve()): _sha256(path) for path in all_sources},
        "test_labels": "stored separately in test_labels.npz",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def _parse_visual_specs(specs: list[str]) -> dict[str, Path] | None:
    if not specs:
        return None
    parsed = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError("--visual must use NAME=PATH")
        name, raw_path = spec.split("=", 1)
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name):
            raise ValueError(f"Invalid visual encoder name: {name}")
        if name in parsed:
            raise ValueError(f"Duplicate visual encoder name: {name}")
        parsed[name] = Path(raw_path).expanduser()
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--visual",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="pid-indexed .pkl, .csv, or .parquet features. Repeat for multiple encoders.",
    )
    parser.add_argument(
        "--aux-dir",
        type=Path,
        help="optional directory with ASR, OCR, caption, and media-statistic CSV files",
    )
    args = parser.parse_args()
    metadata = prepare_smp(
        args.data_dir,
        args.output_dir,
        _parse_visual_specs(args.visual),
        args.aux_dir,
    )
    print(json.dumps(metadata["splits"], indent=2))
    print(f"Prepared data saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
