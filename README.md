# Adaptive Frame Budgeting for Continual Video Learning
### A  proof-of-concept, built toward the  "Next Generation Continual Learning" 

> **Status: proof-of-concept.** This is a scaled-down, single-seed pilot built in one week by two people, not a full study. It is designed to test one specific comparison cleanly rather than to be comprehensive. See [Limitations](#limitations) for exactly what is and isn't covered, and [Full Project Plan](#full-project-plan) for what this would grow into.

## Abstract
Video-based continual learning needs a memory buffer of past examples to fight catastrophic forgetting, but storing whole videos doesn't scale. The common fix — store a fixed number of frames per video — treats an information-dense video and a mostly-static one identically. We test whether allocating a **variable number of stored frames per video**, driven by cheap feature-space diversity and temporal-change signals, preserves accuracy better than a fixed-frame-count baseline at the **same total memory budget**.

## Research Question
> At a fixed total number of stored frames, does adaptive per-video frame allocation (informed by diversity + temporal change) beat a fixed per-video frame count on continual-learning accuracy and forgetting?

## Method (short version)
1. Sample 16 candidate frames/video, encode with a **frozen CLIP ViT-B/32** image encoder (no fine-tuning of the backbone at any point).
2. Score each video: `S(v) = 0.5 * diversity(v) + 0.5 * temporal_change(v)` (mean pairwise and mean consecutive-frame cosine distances, z-normalized per task).
3. Allocate a per-video frame budget via **greedy water-filling** so the total stored frames across all videos exactly matches a fixed global budget.
4. Select frames within each video via **farthest-point (k-center) sampling** on their features.
5. Replay stored frames when training on later tasks; compare against a fixed-K-frames-per-video baseline and a no-memory fine-tuning baseline.

Full derivation and pseudocode: see `docs/design_doc.md` (the Phase 1 design document this project is scoped down from).

## Repository Structure
```
adaptive-video-cl-week/
├── README.md
├── requirements.txt
├── data/
│   └── prepare_subset.py      # [Day 1] builds the 15-class / 3-task split
├── src/
│   ├── features.py            # [Day 1] frozen CLIP feature extraction + caching
│   ├── memory.py               # [Day 3] scoring, water-filling budget, k-center selection
│   ├── cl_loop.py              # [Day 2/4] task loop: finetune / uniform / adaptive modes
│   └── metrics.py              # [Day 2] accuracy, forgetting
├── configs/
│   ├── finetune.yaml
│   ├── uniform.yaml
│   └── adaptive.yaml
├── train.py                    # [Day 4] entry point, reads a config, runs the CL loop
├── evaluate.py                 # [Day 5] loads a run's outputs, prints/generates the results table
└── results/
    ├── table.md                 # filled in on Day 5; "Not yet measured" until then
    └── plots/
```
*(Files not yet implemented are marked with the day they're built; this README is written ahead of the code on purpose, as the contract for what each script must do.)*

## Installation
```bash
git clone <repo-url> adaptive-video-cl-week
cd adaptive-video-cl-week
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Dataset Preparation
Download UCF-101 separately (not redistributed here) and point `--data-root` at the extracted folder (one subdirectory per class).

```bash
python data/prepare_subset.py \
    --data-root /path/to/UCF-101 \
    --output-dir data/splits \
    --num-classes 15 \
    --classes-per-task 5 \
    --train-per-class 15 \
    --test-per-class 5 \
    --seed 0
```
This writes `data/splits/task_manifest.json` and `task_{0,1,2}_{train,test}.csv` — 3 disjoint 5-class tasks, 15 train / 5 test videos per class.

## 2. Feature Extraction
Run once per split file (train and test, for each task). This is the slow step; results are cached, so it only needs to run once.

```bash
for split in data/splits/task_*.csv; do
  python -m src.features \
      --split-csv "$split" \
      --cache-dir cache/features \
      --n-frames 16 \
      --device cuda   # use cpu if no GPU available
done
```
Output: `cache/features/index.csv` (video → cached feature file) plus one `.npy` per video (`n_frames x 512`).

## 3. Training *(Day 2–4)*
```bash
python train.py --config configs/finetune.yaml
python train.py --config configs/uniform.yaml
python train.py --config configs/adaptive.yaml
```
Each config specifies: feature cache location, memory budget (`total_stored_frames`), method (`finetune` / `uniform` / `adaptive`), and a fixed seed. All three configs use the **same total stored-frame budget** for `uniform` and `adaptive` so the comparison is fair.

## 4. Evaluation *(Day 5)*
```bash
python evaluate.py --runs finetune uniform adaptive --output results/table.md
```
Regenerates `results/table.md` and the plots in `results/plots/` directly from logged run outputs — never hand-edited.

## Results
| Method | Total Stored Frames | Final Accuracy | Forgetting |
|---|---:|---:|---:|
| Fine-tuning (no memory) | 0 | Not yet measured | Not yet measured |
| Uniform Replay (fixed K) | *(set on Day 4)* | Not yet measured | Not yet measured |
| Adaptive Replay (ours) | *(same as above)* | Not yet measured | Not yet measured |

No numbers are filled in until `evaluate.py` actually produces them on Day 5.

## Hardware Used / Expected Runtime
- To be filled in with the actual machine used (GPU model, RAM) and observed wall-clock time per stage once run.

## Random Seeds
- Single seed (`seed=0`) throughout. This is a proof-of-concept, not a statistically validated result — a real conclusion would need ≥3 seeds (see Limitations).

## Limitations
- **Single seed, single budget point.** No variance estimate, no budget sweep (10/25/50/75/100%). One clean comparison at one budget, not a full study.
- **3 tasks / 15 classes only**, not the full UCF-101 / vCLIMB protocol.
- **α fixed at 0.5**, not tuned or ablated.
- **No regularization baselines** (EWC/LwF) — only fine-tuning and two replay variants.
- Frozen-backbone diversity reflects CLIP's pretraining notion of "different," not necessarily task-discriminative diversity.
- This scope was chosen deliberately for a 1-week, 2-person build; see the full design doc for what a complete version looks like.

## Full Project Plan
The complete (non-time-boxed) version of this project — full vCLIMB splits, budget sweep, EWC/LwF baselines, multi-seed statistics, signal ablations — is described in `docs/design_doc.md`. This repo is the smallest slice of that plan that still tests the core hypothesis honestly.

## Citation / Related Work
This project builds directly on work from KAUST's IVUL group:
- Villa et al., *vCLIMB: A Novel Video Class Incremental Learning Benchmark*, CVPR 2022 — defines the benchmark protocol and metrics this project's splits are inspired by.
- The group's more recent work on variable per-video memory allocation for video class-incremental learning informed the framing of the research question here; this project is an independent, simplified investigation, not a reproduction of that method.

## License
MIT (see `LICENSE`).
