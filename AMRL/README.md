# AMRL

Reference implementation of **Anchor-Guided Multimodal Residual Learning for Social Media Popularity Prediction**.

AMRL first fits a CatBoost anchor for the global popularity scale. Three branches then learn bounded log-ratio residuals from visual neighbors, lexical statistics, and multimodal neural features. A centered MLP correction performs the final refinement.

## Repository layout

```text
AMRL/
|-- amrl/                 # Model, branches, fusion, and metrics
|-- extract_smp_features.py # Raw video feature extraction
|-- prepare_smp.py        # SMP chronological split and feature preparation
|-- train.py              # Training, validation selection, and test evaluation
|-- tests/                # Split and label-isolation check
`-- requirements.txt
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

FFmpeg must also be available on the system for Whisper audio decoding.

## SMP data example

The example starts from the 4,000 labeled posts released for the SMP Challenge.  Put the following files in one directory:

```text
/path/to/SMP/data/
|-- posts_train.parquet
|-- users_train.parquet
|-- videos_train.parquet
`-- labels_train.parquet
```

Raw videos follow the paths stored in `posts_train.parquet`. A typical layout is:

```text
/path/to/SMP/videos/
`-- train/
    `-- USER.../
        `-- VIDEO....mp4
```

### 1. Extract multimodal features

This command samples eight frames and extracts ViT, CLIP, Whisper ASR, RapidOCR text, BLIP captions, and media properties:

```bash
python extract_smp_features.py \
  --data-dir /path/to/SMP/data \
  --video-root /path/to/SMP/videos \
  --output-dir data/smp_features \
  --device cuda
```

Tasks can be run separately and resumed from their caches:

```bash
python extract_smp_features.py \
  --data-dir /path/to/SMP/data \
  --video-root /path/to/SMP/videos \
  --output-dir data/smp_features \
  --tasks visual caption stats \
  --device cuda
```

The default extraction produces:

```text
data/smp_features/
|-- vit_base_frame8.csv
|-- clip_vitl14_frame8_temporal.csv
|-- video_asr.csv
|-- frame_ocr_rapid.csv
|-- blip_video_captions_f01234567_t24.csv
|-- video_stats.csv
`-- video_file_props.csv
```

### 2. Prepare the chronological split

```bash
python prepare_smp.py \
  --data-dir /path/to/SMP/data \
  --output-dir data/smp \
  --visual vit=data/smp_features/vit_base_frame8.csv \
  --visual clip=data/smp_features/clip_vitl14_frame8_temporal.csv \
  --aux-dir data/smp_features
```

When `--aux-dir` is given, available ASR, OCR, BLIP caption, video-statistic, and file-property CSV files are added automatically. Missing optional files are skipped.

If a precomputed `visual_features.pkl` dictionary is already available in the SMP data directory, the shorter one-encoder example is:

```bash
python prepare_smp.py \
  --data-dir /path/to/SMP/data \
  --output-dir data/smp
```

The processor accepts pid-indexed pickle dictionaries, CSV files, or parquet files. Missing visual vectors are replaced with zeros and their counts are recorded in `metadata.json`. It sorts all labeled posts by `(post_time, pid)` and creates an exact 80/10/10 split. For the released 4,000 labels this gives 3,200 training, 400 validation, and 400 held-out test samples. Numeric transformations, categorical vocabularies, and the visual PCA used by the neural branch are fitted on the training partition. `test.npz` contains no labels. Its labels are stored separately in `test_labels.npz` for evaluation after predictions are fixed.

Prepared files are:

```text
data/smp/
|-- train.npz
|-- valid.npz
|-- test.npz
|-- test_labels.npz
|-- split_manifest.csv
`-- metadata.json
```

### 3. Train AMRL

Run the SMP temporal experiment on a CUDA device:

```bash
python train.py \
  --data-dir data/smp \
  --output-dir runs/smp \
  --device cuda
```

The training protocol is fixed as follows:

1. Forward out-of-fold anchor predictions are produced inside the training partition.
2. Residual targets and label-derived retrieval or lexical statistics use these training OOF rows only.
3. Anchor hyperparameters and the refinement weight are selected on the validation partition.
4. Test predictions are produced before `test_labels.npz` is opened.

The run writes component predictions for validation and test to CSV. It also writes MAPE, SRC, the selected refinement weight, sample counts, and elapsed time to `results.json`.

## Data check

The included check verifies chronological ordering, the 80/10/10 counts, feature alignment, and separation of held-out test labels:

```bash
python -m unittest discover -s tests
```

The repository contains processing and training code only. SMP data and pretrained representations must be obtained from their respective providers.
