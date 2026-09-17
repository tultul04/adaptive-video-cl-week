"""
cl_loop.py

The continual-learning training loop. Supports two methods today:

  - "finetune": no memory at all. Each task is trained using only its
    own 75 training videos. Expect this to forget earlier tasks badly --
    that's the point, it's the control condition.
  - "uniform":  after finishing a task, a FIXED number of frames per
    video (config: frames_per_video) is kept in a memory buffer and
    replayed (mixed into the training set) for every later task.

  - "adaptive" is intentionally NOT implemented yet -- it needs
    src/memory.py (Day 3) to decide a variable number of frames per
    video. Calling run() with method="adaptive" raises NotImplementedError.

All three methods will share this exact file: the model, the optimizer,
and the evaluation code never change. Only what's fed into build_dataset()
differs. That's what makes the eventual 3-way comparison fair.

------------------------------------------------------------------------
PERSON A owns: set_seed, LinearClassifier, train_one_task, evaluate_task
PERSON B owns: MemoryBuffer, select_uniform_frames, the "uniform" branch
                inside run()
Both: load_feature_index, load_split, load_pooled_feature, build_dataset,
      run() are shared plumbing -- read them together before splitting up.
------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.metrics import accuracy, backward_forgetting, final_average_accuracy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared plumbing
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class VideoRecord:
    video_path: str
    label: int
    class_name: str
    task_id: int
    cache_file: str


def load_feature_index(cache_dir: Path) -> dict[str, str]:
    """Reads cache/features/index.csv -> {video_path: cache_file}."""
    index_path = cache_dir / "index.csv"
    if not index_path.exists():
        raise FileNotFoundError(
            f"{index_path} not found. Run src/features.py on all split CSVs first "
            "(see README Step 2)."
        )
    mapping = {}
    with index_path.open() as f:
        for row in csv.DictReader(f):
            mapping[row["video_path"]] = row["cache_file"]
    return mapping


def load_split(split_csv: Path, index: dict[str, str]) -> list[VideoRecord]:
    """Reads a task_*_train.csv or task_*_test.csv and attaches each row's
    cached feature file. Rows whose video failed feature extraction
    (missing from index.csv) are skipped with a warning, not a crash."""
    records = []
    with split_csv.open() as f:
        for row in csv.DictReader(f):
            vp = row["video_path"]
            if vp not in index:
                log.warning("No cached features for %s; skipping (was it dropped during extraction?)", vp)
                continue
            records.append(
                VideoRecord(
                    video_path=vp,
                    label=int(row["label"]),
                    class_name=row["class_name"],
                    task_id=int(row["task_id"]),
                    cache_file=index[vp],
                )
            )
    return records


def load_pooled_feature(cache_dir: Path, cache_file: str, frame_indices: list[int] | None = None) -> np.ndarray:
    """Loads a video's cached (n_frames, 512) CLIP features and mean-pools
    them into a single (512,) vector. If frame_indices is given, only those
    frames are pooled -- this is how a memory-budget constraint is applied."""
    feats = np.load(cache_dir / cache_file)  # (n_frames, 512)
    if frame_indices is not None:
        feats = feats[frame_indices]
    return feats.mean(axis=0)


def build_dataset(
    records: list[VideoRecord],
    cache_dir: Path,
    memory: "MemoryBuffer | None",
) -> tuple[np.ndarray, np.ndarray]:
    """Builds (X, y) for the CURRENT task's full-frame videos, plus
    (for uniform/adaptive) whatever is currently sitting in the memory
    buffer from earlier tasks."""
    X, y = [], []
    for r in records:
        X.append(load_pooled_feature(cache_dir, r.cache_file))
        y.append(r.label)
    if memory is not None:
        for cache_file, label, frame_indices in memory.as_list():
            X.append(load_pooled_feature(cache_dir, cache_file, frame_indices))
            y.append(label)
    if not X:
        raise RuntimeError("Empty dataset -- check that feature extraction actually produced .npy files.")
    return np.stack(X), np.array(y)


# ---------------------------------------------------------------------------
# PERSON A: model + generic training/eval
# ---------------------------------------------------------------------------

class LinearClassifier(nn.Module):
    """A single linear head shared across ALL tasks (class-incremental
    setup): output width = total number of classes across the whole
    experiment, not just the current task's 5 classes."""

    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def train_one_task(
    model: LinearClassifier,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
) -> None:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    n = len(X)

    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb, yb = X_t[idx], y_t[idx]
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        log.info("  epoch %d/%d - loss %.4f", epoch + 1, epochs, total_loss / n)


@torch.no_grad()
def evaluate_task(model: LinearClassifier, X: np.ndarray, y: np.ndarray, device: str) -> float:
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    logits = model(X_t).cpu().numpy()
    return accuracy(logits, y)


# ---------------------------------------------------------------------------
# PERSON B: memory buffer + uniform frame selection
# ---------------------------------------------------------------------------

@dataclass
class MemoryBuffer:
    """Maps video_path -> (cache_file, label, kept_frame_indices).
    Once a video is added, replaying it just means: load its cached
    features, keep only kept_frame_indices, mean-pool, use as one
    training example."""

    entries: dict[str, tuple[str, int, list[int]]] = field(default_factory=dict)

    def add(self, video_path: str, cache_file: str, label: int, frame_indices: list[int]) -> None:
        self.entries[video_path] = (cache_file, label, frame_indices)

    def total_frames(self) -> int:
        return sum(len(idx) for _, _, idx in self.entries.values())

    def as_list(self) -> list[tuple[str, int, list[int]]]:
        return list(self.entries.values())


def select_uniform_frames(n_available: int, k: int) -> list[int]:
    """Evenly-spaced indices among the n_available candidate frames --
    e.g. n_available=16, k=5 -> [0, 4, 8, 12, 15]. This is the "fixed K
    frames per video, no matter the content" baseline strategy."""
    k = min(k, n_available)
    if k <= 0:
        return []
    return sorted(set(int(i) for i in np.linspace(0, n_available - 1, k)))


# ---------------------------------------------------------------------------
# Main loop (shared)
# ---------------------------------------------------------------------------

def run(config: dict) -> dict:
    set_seed(config["seed"])
    device = "cuda" if (config.get("device") == "cuda" and torch.cuda.is_available()) else "cpu"

    cache_dir = Path(config["feature_cache_dir"])
    splits_dir = Path(config["splits_dir"])
    num_tasks = config["num_tasks"]
    method = config["method"]

    if method not in ("finetune", "uniform"):
        raise NotImplementedError(
            f"Method '{method}' isn't implemented yet. "
            "'adaptive' needs src/memory.py, which lands on Day 3."
        )

    index = load_feature_index(cache_dir)
    manifest = json.loads((splits_dir / "task_manifest.json").read_text())
    num_classes = len(manifest["classes"])
    feat_dim = int(np.load(cache_dir / next(iter(index.values()))).shape[1])
    log.info("num_classes=%d feat_dim=%d device=%s", num_classes, feat_dim, device)

    model = LinearClassifier(feat_dim, num_classes).to(device)
    memory = MemoryBuffer() if method == "uniform" else None
    frames_per_video = config.get("frames_per_video")

    test_sets: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    R: list[list[float]] = []

    for t in range(num_tasks):
        log.info("=== Task %d (%s) ===", t, method)
        train_records = load_split(splits_dir / f"task_{t}_train.csv", index)
        test_records = load_split(splits_dir / f"task_{t}_test.csv", index)

        X_train, y_train = build_dataset(train_records, cache_dir, memory)
        log.info("Training set size (current task + replay): %d", len(X_train))
        train_one_task(
            model, X_train, y_train,
            epochs=config["epochs_per_task"], lr=config["lr"],
            batch_size=config["batch_size"], device=device,
        )

        X_test, y_test = build_dataset(test_records, cache_dir, None)
        test_sets[t] = (X_test, y_test)

        row = []
        for j in range(t + 1):
            Xj, yj = test_sets[j]
            acc = evaluate_task(model, Xj, yj, device)
            row.append(acc)
            log.info("  accuracy on task %d test set: %.4f", j, acc)
        R.append(row)

        if method == "uniform":
            for r in train_records:
                n_available = int(np.load(cache_dir / r.cache_file).shape[0])
                kept = select_uniform_frames(n_available, frames_per_video)
                memory.add(r.video_path, r.cache_file, r.label, kept)
            log.info(
                "Memory buffer: %d videos, %d total stored frames",
                len(memory.entries), memory.total_frames(),
            )

    final_acc = final_average_accuracy(R)
    forgetting = backward_forgetting(R)
    log.info("FINAL Average Accuracy: %.4f", final_acc)
    log.info("FINAL Backward Forgetting: %.4f", forgetting)

    return {
        "method": method,
        "accuracy_matrix": R,
        "final_average_accuracy": final_acc,
        "backward_forgetting": forgetting,
        "total_stored_frames": memory.total_frames() if memory else 0,
    }