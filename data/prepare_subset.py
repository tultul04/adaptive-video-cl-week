"""
prepare_subset.py

Builds the small, fixed continual-learning subset used for the 1-week
proof-of-concept: N_CLASSES classes from UCF-101, split into disjoint
tasks of CLASSES_PER_TASK classes each, with a train/test split per class.

Expected input layout (standard UCF-101 extraction):

    <data_root>/
        ApplyEyeMakeup/
            v_ApplyEyeMakeup_g01_c01.avi
            ...
        Archery/
            ...
        ...

Output:
    <output_dir>/task_manifest.json          # class -> task_id mapping, for reference
    <output_dir>/task_{i}_train.csv          # video_path,label,class_name,task_id
    <output_dir>/task_{i}_test.csv

Usage:
    python data/prepare_subset.py \
        --data-root /path/to/UCF-101 \
        --output-dir data/splits \
        --num-classes 15 \
        --classes-per-task 5 \
        --train-per-class 15 \
        --test-per-class 5 \
        --seed 0
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

VIDEO_EXTS = {".avi", ".mp4", ".mkv", ".mov"}


@dataclass
class ClassSplit:
    class_name: str
    task_id: int
    train_videos: list[Path]
    test_videos: list[Path]


def list_class_dirs(data_root: Path) -> list[Path]:
    dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
    if not dirs:
        raise FileNotFoundError(
            f"No class subdirectories found under {data_root}. "
            "Expected UCF-101-style layout: <data_root>/<ClassName>/<video files>."
        )
    return dirs


def list_videos(class_dir: Path) -> list[Path]:
    videos = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    return videos


def build_subset(
    data_root: Path,
    num_classes: int,
    classes_per_task: int,
    train_per_class: int,
    test_per_class: int,
    seed: int,
) -> list[ClassSplit]:
    if num_classes % classes_per_task != 0:
        raise ValueError(
            f"num_classes ({num_classes}) must be divisible by classes_per_task "
            f"({classes_per_task}) so tasks are balanced."
        )

    rng = random.Random(seed)
    all_class_dirs = list_class_dirs(data_root)

    # Only keep classes that actually have enough videos for the requested split.
    usable = []
    needed = train_per_class + test_per_class
    for class_dir in all_class_dirs:
        videos = list_videos(class_dir)
        if len(videos) >= needed:
            usable.append((class_dir, videos))
        else:
            log.warning(
                "Skipping class '%s': has %d videos, need >= %d",
                class_dir.name, len(videos), needed,
            )

    if len(usable) < num_classes:
        raise ValueError(
            f"Only {len(usable)} classes have enough videos (need {needed} each); "
            f"requested {num_classes}. Lower train/test-per-class or num-classes."
        )

    rng.shuffle(usable)
    chosen = usable[:num_classes]

    splits: list[ClassSplit] = []
    for idx, (class_dir, videos) in enumerate(chosen):
        rng.shuffle(videos)
        train_videos = videos[:train_per_class]
        test_videos = videos[train_per_class:train_per_class + test_per_class]
        task_id = idx // classes_per_task
        splits.append(
            ClassSplit(
                class_name=class_dir.name,
                task_id=task_id,
                train_videos=train_videos,
                test_videos=test_videos,
            )
        )
        log.info(
            "Class '%s' -> task %d (%d train, %d test videos)",
            class_dir.name, task_id, len(train_videos), len(test_videos),
        )

    return splits


def write_csv(rows: list[tuple[str, int, str, int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "label", "class_name", "task_id"])
        writer.writerows(rows)
    log.info("Wrote %d rows -> %s", len(rows), path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True, help="Path to UCF-101 root (class subdirectories).")
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--num-classes", type=int, default=15)
    parser.add_argument("--classes-per-task", type=int, default=5)
    parser.add_argument("--train-per-class", type=int, default=15)
    parser.add_argument("--test-per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    splits = build_subset(
        data_root=args.data_root,
        num_classes=args.num_classes,
        classes_per_task=args.classes_per_task,
        train_per_class=args.train_per_class,
        test_per_class=args.test_per_class,
        seed=args.seed,
    )

    # Assign a stable integer label per class, ordered by (task_id, class_name)
    ordered = sorted(splits, key=lambda s: (s.task_id, s.class_name))
    label_of = {s.class_name: i for i, s in enumerate(ordered)}

    manifest = {
        "seed": args.seed,
        "num_classes": args.num_classes,
        "classes_per_task": args.classes_per_task,
        "classes": {
            s.class_name: {"label": label_of[s.class_name], "task_id": s.task_id}
            for s in splits
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "task_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Wrote manifest -> %s", args.output_dir / "task_manifest.json")

    num_tasks = args.num_classes // args.classes_per_task
    for task_id in range(num_tasks):
        task_splits = [s for s in splits if s.task_id == task_id]

        train_rows = [
            (str(v), label_of[s.class_name], s.class_name, task_id)
            for s in task_splits for v in s.train_videos
        ]
        test_rows = [
            (str(v), label_of[s.class_name], s.class_name, task_id)
            for s in task_splits for v in s.test_videos
        ]

        write_csv(train_rows, args.output_dir / f"task_{task_id}_train.csv")
        write_csv(test_rows, args.output_dir / f"task_{task_id}_test.csv")

    log.info("Done. %d tasks written to %s", num_tasks, args.output_dir)


if __name__ == "__main__":
    main()
