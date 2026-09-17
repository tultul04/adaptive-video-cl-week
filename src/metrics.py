"""
metrics.py

Standard class-incremental continual-learning metrics, computed from an
accuracy matrix R, where:

    R[i][j] = accuracy on task j's TEST set, using the model exactly as
              it existed right after finishing training on task i.
              Only defined for j <= i (you can't evaluate on a task's
              test set before that task has been trained on).

Example for 3 tasks (rows = "after training on task i", growing by one
entry each time):
    R = [
        [0.91],                   # after task 0: acc on task 0
        [0.55, 0.88],             # after task 1: acc on task 0, task 1
        [0.40, 0.52, 0.85],       # after task 2: acc on task 0, task 1, task 2
    ]

From this we derive:
  - Final Average Accuracy: mean of the LAST row (how good is the model,
    at the very end, on every task it has ever seen).
  - Backward Forgetting (BWF): for each task except the last, how much
    accuracy dropped from its best-ever measured point down to the final
    point, averaged across those tasks. Higher = worse (more forgetting).
"""

from __future__ import annotations

import numpy as np


def accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    """logits: (n, num_classes), labels: (n,) integer class ids."""
    preds = logits.argmax(axis=1)
    return float((preds == labels).mean())


def final_average_accuracy(R: list[list[float]]) -> float:
    """Mean accuracy across all tasks, measured at the very end of training."""
    last_row = R[-1]
    return float(np.mean(last_row))


def backward_forgetting(R: list[list[float]]) -> float:
    """
    For each task j (except the final one), forgetting is:
        max(R[j][j], R[j+1][j], ..., R[T-2][j]) - R[T-1][j]
    i.e. the biggest drop from that task's best-ever recorded accuracy
    down to its accuracy at the very end of the whole run. Averaged over
    all such tasks.

    Returns 0.0 if there's only one task (nothing can be "forgotten" yet).
    """
    T = len(R)
    if T < 2:
        return 0.0

    forgettings = []
    for j in range(T - 1):
        # R[i][j] is defined for i in [j, T-1]; "best-ever before the end"
        # excludes the final measurement, which is what we're comparing against.
        history = [R[i][j] for i in range(j, T - 1)]
        best_before_end = max(history) if history else R[j][j]
        final = R[T - 1][j]
        forgettings.append(best_before_end - final)

    return float(np.mean(forgettings))
