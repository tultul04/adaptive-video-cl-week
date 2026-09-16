"""
features.py

Extracts N candidate frames per video (uniformly in time) and encodes them
with a frozen CLIP image encoder. Features are cached to disk as .npy files
so every downstream method (fine-tuning, uniform replay, adaptive replay)
reuses exactly the same frame features -- this is what makes the comparison
between methods fair (Section 5/7 of the design doc).

Requires: opencv-python, open_clip_torch, torch, numpy

Usage:
    python -m src.features \
        --split-csv data/splits/task_0_train.csv \
        --cache-dir cache/features \
        --n-frames 16 \
        --device cuda   # or cpu

Output per video: cache/features/<video_stem>__<hash>.npy, shape (n_frames, feat_dim)
A companion index file cache/features/index.csv maps video_path -> cache file,
so later steps never need to touch raw video files again.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def sample_frame_indices(total_frames: int, n_frames: int) -> list[int]:
    """Uniformly spaced frame indices covering the whole video."""
    if total_frames <= 0:
        raise ValueError("Video reports zero frames; file may be corrupt.")
    if total_frames <= n_frames:
        return list(range(total_frames))
    # Evenly spaced, inclusive of first frame, avoiding the very last frame
    # (often a duplicate/black frame in some UCF-101 encodings).
    step = total_frames / n_frames
    return [int(i * step) for i in range(n_frames)]


def read_frames(video_path: Path, n_frames: int) -> np.ndarray:
    """Returns an array of shape (k, H, W, 3) in RGB, k <= n_frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = set(sample_frame_indices(total, n_frames))

    frames = []
    idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if idx in indices:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        idx += 1
    cap.release()

    if not frames:
        raise IOError(f"No frames decoded from: {video_path}")
    return np.stack(frames)


def cache_key(video_path: Path, n_frames: int) -> str:
    h = hashlib.sha1(f"{video_path.resolve()}::{n_frames}".encode()).hexdigest()[:10]
    return f"{video_path.stem}__{h}"


class ClipEncoder:
    """Thin wrapper around a frozen CLIP image encoder. No gradients, ever."""

    def __init__(self, device: str = "cpu", model_name: str = "ViT-B-32", pretrained: str = "openai"):
        import open_clip  # imported lazily so the rest of the module has no hard dependency

        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode(self, frames_rgb: np.ndarray) -> np.ndarray:
        """frames_rgb: (k, H, W, 3) uint8 -> returns (k, feat_dim) float32, L2-normalized."""
        from PIL import Image

        batch = torch.stack(
            [self.preprocess(Image.fromarray(f)) for f in frames_rgb]
        ).to(self.device)
        feats = self.model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)


def extract_and_cache(
    split_csv: Path,
    cache_dir: Path,
    n_frames: int,
    device: str,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "index.csv"
    already_cached = set()
    if index_path.exists():
        with index_path.open() as f:
            already_cached = {row["video_path"] for row in csv.DictReader(f)}

    encoder = ClipEncoder(device=device)

    with split_csv.open() as f:
        rows = list(csv.DictReader(f))
    log.info("Loaded %d videos from %s", len(rows), split_csv)

    index_rows = []
    for i, row in enumerate(rows, 1):
        video_path = Path(row["video_path"])
        if row["video_path"] in already_cached:
            log.info("[%d/%d] already cached, skipping: %s", i, len(rows), video_path.name)
            continue
        try:
            frames = read_frames(video_path, n_frames)
            feats = encoder.encode(frames)
        except (IOError, ValueError) as e:
            log.error("Skipping %s: %s", video_path, e)
            continue

        key = cache_key(video_path, n_frames)
        out_path = cache_dir / f"{key}.npy"
        np.save(out_path, feats)
        index_rows.append(
            {
                "video_path": row["video_path"],
                "label": row["label"],
                "class_name": row["class_name"],
                "task_id": row["task_id"],
                "cache_file": out_path.name,
                "n_frames_extracted": feats.shape[0],
            }
        )
        log.info("[%d/%d] cached %s -> %s (%d frames)", i, len(rows), video_path.name, out_path.name, feats.shape[0])

    write_header = not index_path.exists()
    with index_path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["video_path", "label", "class_name", "task_id", "cache_file", "n_frames_extracted"]
        )
        if write_header:
            writer.writeheader()
        writer.writerows(index_rows)
    log.info("Appended %d new entries -> %s", len(index_rows), index_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-csv", type=Path, required=True, help="A task_*_train.csv or task_*_test.csv file.")
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/features"))
    parser.add_argument("--n-frames", type=int, default=16, help="Candidate frames sampled per video.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    extract_and_cache(args.split_csv, args.cache_dir, args.n_frames, args.device)


if __name__ == "__main__":
    main()
