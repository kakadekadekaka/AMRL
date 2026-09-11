from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from amrl import AMRL, AMRLConfig, Split, metrics


LEXICAL_SOURCES = ("content", "tags", "music", "asr", "ocr", "caption")


def load_split(path: Path) -> tuple[np.ndarray, Split]:
    with np.load(path, allow_pickle=False) as data:
        visual = {
            key.removeprefix("visual__"): data[key].astype(np.float32)
            for key in sorted(data.files)
            if key.startswith("visual__")
        }
        lexical_columns = {
            source: data[f"lexical__{source}"].astype(str)
            for source in LEXICAL_SOURCES
        }
        lexical = [
            {source: lexical_columns[source][row] for source in LEXICAL_SOURCES}
            for row in range(len(data["pid"]))
        ]
        split = Split(
            anchor=data["anchor"].astype(np.float32),
            visual=visual,
            neural=data["neural"].astype(np.float32),
            lexical=lexical,
            y=data["y"].astype(np.float64) if "y" in data else None,
            time=data["time"],
        )
        return data["pid"].astype(str), split


def _scored(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    return {"eval_samples": len(y), **metrics(y, prediction)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--oof-splits", type=int, default=4)
    args = parser.parse_args()

    started = time.monotonic()
    train_ids, train = load_split(args.data_dir / "train.npz")
    valid_ids, valid = load_split(args.data_dir / "valid.npz")
    test_ids, test = load_split(args.data_dir / "test.npz")
    if test.y is not None:
        raise ValueError("test.npz must not contain labels")

    model = AMRL(AMRLConfig(
        temporal=True,
        oof_splits=args.oof_splits,
        threads=args.threads,
        device=args.device,
    ))
    validation_metrics = model.fit(train, valid)
    test_components = model.predict_components(test)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"pid": valid_ids, **model.validation_components_}).to_csv(
        args.output_dir / "validation_predictions.csv", index=False
    )
    pd.DataFrame({"pid": test_ids, **test_components}).to_csv(
        args.output_dir / "test_predictions.csv", index=False
    )

    result = {
        "protocol": "fit on chronological train, select on valid, evaluate held-out test once",
        "device": args.device,
        "train_samples": len(train_ids),
        "valid_samples": len(valid_ids),
        "test_samples": len(test_ids),
        "selected_alpha": float(model.alpha_),
        "validation": {
            name: {"eval_samples": len(valid_ids), **values}
            for name, values in validation_metrics.items()
        },
    }

    labels_path = args.data_dir / "test_labels.npz"
    if labels_path.is_file():
        with np.load(labels_path, allow_pickle=False) as labels:
            label_ids = labels["pid"].astype(str)
            if not np.array_equal(label_ids, test_ids):
                raise ValueError("test_labels.npz is not aligned with test.npz")
            result["test"] = _scored(labels["y"], test_components["final"])
    result["elapsed_seconds"] = time.monotonic() - started
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
