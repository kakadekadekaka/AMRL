"""Extract AMRL input features from the labeled SMP videos.

The script supports visual embeddings, speech transcription, frame OCR,
frame captions, and media properties. Each task writes a resumable cache and
the CSV files consumed by prepare_smp.py.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


TASKS = ("visual", "asr", "ocr", "caption", "stats")


def _media_index(data_dir: Path, video_root: Path) -> pd.DataFrame:
    posts = pd.read_parquet(data_dir / "posts_train.parquet")
    videos = pd.read_parquet(data_dir / "videos_train.parquet")
    if posts["pid"].duplicated().any() or videos["pid"].duplicated().any():
        raise ValueError("posts and videos must contain unique pid values")
    table = posts[["pid", "uid", "video_path"]].merge(
        videos[["pid", "uid", "vid"]],
        on=["pid", "uid"],
        how="left",
        validate="one_to_one",
    )
    if table["vid"].isna().any():
        raise ValueError("videos_train.parquet failed to align with posts_train.parquet")

    def resolve(row: pd.Series) -> Path:
        listed = Path(str(row["video_path"]))
        candidates = [
            listed,
            video_root / listed,
            video_root / "train" / str(row["uid"]) / f"{row['vid']}.mp4",
        ]
        return next((path for path in candidates if path.is_file()), candidates[1])

    table["resolved_path"] = [resolve(row) for _, row in table.iterrows()]
    return table


def _sample_frames(path: Path, count: int, rgb: bool) -> list[np.ndarray]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        capture.release()
        return []
    frames = []
    for index in np.linspace(0, total - 1, count, dtype=int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        success, frame = capture.read()
        if success:
            if rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    capture.release()
    if frames:
        frames.extend([frames[-1]] * (count - len(frames)))
    return frames


def _device(requested: str):
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else
                        "cpu" if requested == "auto" else requested)


def _load_cache(path: Path, settings: dict) -> dict:
    if not path.is_file():
        return {}
    with path.open("rb") as handle:
        saved = pickle.load(handle)
    if saved.get("settings") != settings:
        raise ValueError(f"Cached settings differ. Remove {path} before rerunning.")
    return saved["values"]


def _save_cache(path: Path, settings: dict, values: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump({"settings": settings, "values": values}, handle)
    temporary.replace(path)


def _run_cached(
    media: pd.DataFrame,
    cache_path: Path,
    settings: dict,
    extract: Callable[[Path], object],
    label: str,
) -> dict:
    from tqdm import tqdm

    cache = _load_cache(cache_path, settings)
    pending = media[~media["pid"].astype(str).isin(cache)]
    for number, row in enumerate(
        tqdm(pending.itertuples(index=False), total=len(pending), desc=label), 1
    ):
        path = Path(row.resolved_path)
        if not path.is_file():
            cache[str(row.pid)] = None
        else:
            try:
                cache[str(row.pid)] = extract(path)
            except Exception as error:
                print(f"{label} failed for {row.pid}: {error}")
                cache[str(row.pid)] = None
        if number % 50 == 0:
            _save_cache(cache_path, settings, cache)
    _save_cache(cache_path, settings, cache)
    return cache


def _write_matrix(
    path: Path,
    pids: list[str],
    cache: dict,
    key: str,
    prefix: str,
) -> None:
    available = [entry[key] for entry in cache.values() if entry is not None]
    if not available:
        raise RuntimeError(f"No features were extracted for {path.name}")
    dimension = len(available[0])
    zero = np.zeros(dimension, dtype=np.float32)
    matrix = np.stack([
        cache.get(pid, {}).get(key, zero) if cache.get(pid) is not None else zero
        for pid in pids
    ])
    frame = pd.DataFrame(matrix, columns=[f"{prefix}_{index}" for index in range(dimension)])
    frame.insert(0, "pid", pids)
    frame.to_csv(path, index=False)
    print(f"Saved {path} with shape {matrix.shape}")


def extract_visual(media: pd.DataFrame, output_dir: Path, args) -> None:
    import timm
    import torch
    from PIL import Image
    from timm.data import create_transform, resolve_model_data_config
    from transformers import CLIPImageProcessor, CLIPVisionModel

    device = _device(args.device)
    vit = timm.create_model(args.vit_model, pretrained=True, num_classes=0).to(device).eval()
    vit_transform = create_transform(
        **resolve_model_data_config(vit), is_training=False
    )
    clip_processor = CLIPImageProcessor.from_pretrained(args.clip_model)
    clip = CLIPVisionModel.from_pretrained(args.clip_model).to(device).eval()
    settings = {
        "frames": args.frames,
        "vit_model": args.vit_model,
        "clip_model": args.clip_model,
    }

    @torch.inference_mode()
    def encode(path: Path) -> dict | None:
        frames = _sample_frames(path, args.frames, rgb=True)
        if not frames:
            return None
        images = [Image.fromarray(frame) for frame in frames]
        vit_batch = torch.stack([vit_transform(image) for image in images]).to(device)
        vit_features = vit(vit_batch)
        if vit_features.ndim == 3:
            vit_features = vit_features[:, 0]
        clip_batch = clip_processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        clip_features = clip(pixel_values=clip_batch).pooler_output
        clip_features = torch.nn.functional.normalize(clip_features, dim=1)
        return {
            "vit": vit_features.float().cpu().numpy().reshape(-1),
            "clip": clip_features.float().cpu().numpy().reshape(-1),
        }

    cache = _run_cached(
        media, output_dir / ".visual_cache.pkl", settings, encode, "Visual"
    )
    pids = media["pid"].astype(str).tolist()
    _write_matrix(output_dir / "vit_base_frame8.csv", pids, cache, "vit", "vit")
    _write_matrix(
        output_dir / "clip_vitl14_frame8_temporal.csv", pids, cache, "clip", "clip"
    )


def extract_asr(media: pd.DataFrame, output_dir: Path, args) -> None:
    import whisper

    device = _device(args.device)
    model = whisper.load_model(args.whisper_model, device=str(device))
    settings = {"model": args.whisper_model, "audio_seconds": args.audio_seconds}

    def transcribe(path: Path) -> dict:
        audio = whisper.load_audio(str(path))[: args.audio_seconds * 16000]
        if len(audio) < 8000:
            return {"asr_text": "", "asr_language": "no_audio", "asr_char_count": 0}
        result = model.transcribe(audio, fp16=device.type == "cuda")
        text = str(result.get("text", "")).strip()
        return {
            "asr_text": text,
            "asr_language": str(result.get("language", "unknown")),
            "asr_char_count": len(text),
        }

    cache = _run_cached(media, output_dir / ".asr_cache.pkl", settings, transcribe, "ASR")
    empty = {"asr_text": "", "asr_language": "", "asr_char_count": 0}
    rows = [
        {"pid": str(pid), **(cache.get(str(pid)) or empty)} for pid in media["pid"]
    ]
    pd.DataFrame(rows).to_csv(output_dir / "video_asr.csv", index=False)


def extract_ocr(media: pd.DataFrame, output_dir: Path, args) -> None:
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()
    settings = {"frames": args.ocr_frames}

    def recognize(path: Path) -> dict:
        texts = []
        for frame in _sample_frames(path, args.ocr_frames, rgb=False):
            result, _ = engine(frame)
            if result:
                texts.extend(str(item[1]).strip() for item in result if str(item[1]).strip())
        text = " | ".join(texts)
        return {"ocr_text": text, "ocr_raw_text_count": len(text)}

    cache = _run_cached(media, output_dir / ".ocr_cache.pkl", settings, recognize, "OCR")
    empty = {"ocr_text": "", "ocr_raw_text_count": 0}
    rows = [
        {"pid": str(pid), **(cache.get(str(pid)) or empty)} for pid in media["pid"]
    ]
    pd.DataFrame(rows).to_csv(output_dir / "frame_ocr_rapid.csv", index=False)


def extract_captions(media: pd.DataFrame, output_dir: Path, args) -> None:
    import torch
    from PIL import Image
    from transformers import BlipForConditionalGeneration, BlipProcessor

    device = _device(args.device)
    processor = BlipProcessor.from_pretrained(args.caption_model)
    model = BlipForConditionalGeneration.from_pretrained(args.caption_model).to(device).eval()
    settings = {
        "frames": args.frames,
        "model": args.caption_model,
        "max_new_tokens": args.caption_tokens,
    }

    @torch.inference_mode()
    def caption(path: Path) -> list[str] | None:
        frames = _sample_frames(path, args.frames, rgb=True)
        if not frames:
            return None
        inputs = processor(
            images=[Image.fromarray(frame) for frame in frames], return_tensors="pt"
        ).to(device)
        output = model.generate(**inputs, max_new_tokens=args.caption_tokens)
        return processor.batch_decode(output, skip_special_tokens=True)

    cache = _run_cached(
        media, output_dir / ".caption_cache.pkl", settings, caption, "Caption"
    )
    rows = []
    for pid in media["pid"].astype(str):
        captions = cache.get(pid) or [""] * args.frames
        rows.append({
            "pid": pid,
            **{f"blip_caption_{index}": captions[index] for index in range(args.frames)},
        })
    pd.DataFrame(rows).to_csv(
        output_dir / "blip_video_captions_f01234567_t24.csv", index=False
    )


def extract_stats(media: pd.DataFrame, output_dir: Path) -> None:
    import cv2
    from tqdm import tqdm

    statistics = []
    properties = []
    for row in tqdm(media.itertuples(index=False), total=len(media), desc="Stats"):
        path = Path(row.resolved_path)
        exists = path.is_file()
        properties.append({
            "pid": str(row.pid),
            "file_size_bytes": path.stat().st_size if exists else 0,
            "file_exists": exists,
        })
        capture = cv2.VideoCapture(str(path)) if exists else None
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture else 0
        fps = float(capture.get(cv2.CAP_PROP_FPS)) if capture else np.nan
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if capture else 0
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if capture else 0
        if capture:
            capture.release()
        statistics.append({
            "pid": str(row.pid),
            "duration": frames / fps if fps > 0 else np.nan,
            "fps": fps,
            "width": width,
            "height": height,
            "aspect_ratio": width / height if height > 0 else np.nan,
            "total_frames": frames,
        })
    pd.DataFrame(statistics).to_csv(output_dir / "video_stats.csv", index=False)
    pd.DataFrame(properties).to_csv(output_dir / "video_file_props.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--ocr-frames", type=int, default=4)
    parser.add_argument("--audio-seconds", type=int, default=30)
    parser.add_argument("--caption-tokens", type=int, default=24)
    parser.add_argument("--vit-model", default="vit_base_patch16_224")
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--caption-model", default="Salesforce/blip-image-captioning-base")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    media = _media_index(args.data_dir, args.video_root)
    print(f"Found {len(media)} posts and {media['resolved_path'].map(Path.is_file).sum()} videos")
    actions = {
        "visual": lambda: extract_visual(media, args.output_dir, args),
        "asr": lambda: extract_asr(media, args.output_dir, args),
        "ocr": lambda: extract_ocr(media, args.output_dir, args),
        "caption": lambda: extract_captions(media, args.output_dir, args),
        "stats": lambda: extract_stats(media, args.output_dir),
    }
    for task in args.tasks:
        actions[task]()


if __name__ == "__main__":
    main()
