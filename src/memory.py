"""
memory.py

Adaptive replay memory selection.

This file decides:
1. How diverse/changing the frames of each video are.
2. How many frames each video should get.
3. Which specific frames should be stored.
"""

import numpy as np
def diversity_score(features: np.ndarray) -> float:
    """
    Calculate the diversity of the candidate frames in one video.

    features has shape:
        (number_of_frames, feature_dimension)

    In our project this will usually be:
        (16, 512)
    """

    # If there are fewer than 2 frames, there is nothing to compare.
    if len(features) < 2:
        return 0.0

    # CLIP features from features.py are already L2-normalized.
    # Therefore, the dot product between two feature vectors
    # gives us their cosine similarity.
    similarity_matrix = features @ features.T

    # Convert cosine similarity into cosine distance.
    # Similar frames -> distance close to 0.
    # Different frames -> larger distance.
    distance_matrix = 1.0 - similarity_matrix

    # We only want each unique pair of frames once.
    # k=1 excludes the diagonal and takes the upper triangle.
    row_indices, col_indices = np.triu_indices(len(features), k=1)

    pairwise_distances = distance_matrix[row_indices, col_indices]

    # Diversity is the average distance between all frame pairs.
    return float(np.mean(pairwise_distances))



def temporal_score(features: np.ndarray) -> float:
    """
    Calculate how much the video changes over time.

    Instead of comparing every possible pair of frames,
    we compare only consecutive frames:
    frame 1 -> frame 2, frame 2 -> frame 3, etc.
    """

    # We need at least two frames to measure change over time.
    if len(features) < 2:
        return 0.0

    # Compare each frame with the frame immediately after it.
    # Since CLIP features are already L2-normalized,
    # dot product gives cosine similarity.
    consecutive_similarities = np.sum(
        features[:-1] * features[1:],
        axis=1
    )

    # Convert cosine similarity to cosine distance.
    consecutive_distances = 1.0 - consecutive_similarities

    # Average all consecutive-frame distances.
    return float(np.mean(consecutive_distances))


def z_normalize(values: np.ndarray) -> np.ndarray:
    """
    Z-normalize a collection of scores.

    This converts the scores into values representing how far
    each score is above or below the average score.
    """

    values = np.asarray(values, dtype=np.float32)

    mean = np.mean(values)
    std = np.std(values)

    # If every value is the same, standard deviation is 0.
    # Dividing by 0 would be invalid, so return all zeros.
    if std < 1e-8:
        return np.zeros_like(values)

    return (values - mean) / std

def combined_scores(
    diversity_scores: np.ndarray,
    temporal_scores: np.ndarray
) -> np.ndarray:
    """
    Combine normalized diversity and temporal scores.

    Each component receives equal weight:
    50% diversity + 50% temporal change.
    """

    # Normalize diversity scores across all videos.
    normalized_diversity = z_normalize(diversity_scores)

    # Normalize temporal scores across all videos.
    normalized_temporal = z_normalize(temporal_scores)

    # Give both measurements equal importance.
    scores = 0.5 * normalized_diversity + 0.5 * normalized_temporal

    return scores

def allocate_budget(
    scores: np.ndarray,
    total_budget: int,
    b_min: int = 2,
    b_max: int = 10,
) -> np.ndarray:
    """
    Allocate an integer number of frames to each video.

    Every video receives at least b_min frames and at most b_max frames.
    The total number of allocated frames must exactly equal total_budget.

    Extra frames are distributed greedily according to the video's score.
    """

    scores = np.asarray(scores, dtype=np.float32)
    num_videos = len(scores)

    if num_videos == 0:
        if total_budget != 0:
            raise ValueError("Cannot allocate a non-zero budget to zero videos.")
        return np.array([], dtype=int)

    # Check whether the requested budget is actually possible.
    minimum_required = num_videos * b_min
    maximum_allowed = num_videos * b_max

    if total_budget < minimum_required:
        raise ValueError(
            f"Budget {total_budget} is too small. "
            f"At least {minimum_required} frames are required."
        )

    if total_budget > maximum_allowed:
        raise ValueError(
            f"Budget {total_budget} is too large. "
            f"At most {maximum_allowed} frames can be allocated."
        )

    # Every video starts with the minimum number of frames.
    allocation = np.full(num_videos, b_min, dtype=int)

    remaining = total_budget - minimum_required

    # Rank videos from highest score to lowest score.
    # mergesort keeps ties deterministic.
    priority = np.argsort(-scores, kind="mergesort")

    # Greedy water-filling:
    # repeatedly make one pass from highest score to lowest score,
    # giving each eligible video one additional frame.
    while remaining > 0:
        allocated_this_round = False

        for video_index in priority:
            if remaining == 0:
                break

            if allocation[video_index] < b_max:
                allocation[video_index] += 1
                remaining -= 1
                allocated_this_round = True

        # Defensive check: normally impossible because we validated
        # total_budget <= num_videos * b_max above.
        if not allocated_this_round:
            raise RuntimeError("Unable to allocate the remaining frame budget.")

    return allocation

def k_center_select(features: np.ndarray, k: int) -> list[int]:
    """
    Select k representative frames using greedy farthest-point
    (k-center) sampling.

    Returns the indices of the selected candidate frames.
    """

    features = np.asarray(features, dtype=np.float32)
    num_frames = len(features)

    if k <= 0:
        return []

    if num_frames == 0:
        return []

    if k >= num_frames:
        return list(range(num_frames))

    # Normalize defensively so dot products represent cosine similarity.
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = features / np.clip(norms, 1e-12, None)

    # Start with the frame closest to the video's mean feature.
    centroid = np.mean(normalized, axis=0)
    centroid_norm = np.linalg.norm(centroid)

    if centroid_norm > 1e-12:
        centroid = centroid / centroid_norm
        first_index = int(np.argmax(normalized @ centroid))
    else:
        first_index = 0

    selected = [first_index]

    # Distance from every candidate frame to the first selected center.
    similarities = normalized @ normalized[first_index]
    min_distances = 1.0 - similarities

    while len(selected) < k:

        # Never select an already-selected frame again.
        min_distances[selected] = -np.inf

        # Choose the frame farthest from its nearest selected center.
        next_index = int(np.argmax(min_distances))
        selected.append(next_index)

        # Calculate distance from every frame to the new center.
        similarities = normalized @ normalized[next_index]
        new_distances = 1.0 - similarities

        # For each candidate, remember its distance to its nearest
        # selected center.
        min_distances = np.minimum(min_distances, new_distances)

    return selected


if __name__ == "__main__":

    # Test 1:
    # All three frames have the same feature vector.
    # Therefore, their diversity should be approximately 0.
    similar_frames = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0]
    ])

    print("Similar frames diversity:", diversity_score(similar_frames))


    # Test 2:
    # These frames point in different directions,
    # so their diversity should be higher.
    different_frames = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0]
    ])

    print("Different frames diversity:", diversity_score(different_frames))
    # Test temporal score
    print("Similar frames temporal:", temporal_score(similar_frames))
    print("Different frames temporal:", temporal_score(different_frames))

    # Test Z-normalization
    test_scores = np.array([0.2, 0.5, 0.8])
    print("Original scores:", test_scores)
    print("Z-normalized scores:", z_normalize(test_scores))

    # Test combined diversity + temporal score
    test_diversity = np.array([0.2, 0.5, 0.8])
    test_temporal = np.array([0.1, 0.7, 0.4])

    print("Combined scores:", combined_scores(test_diversity, test_temporal))

        # Test budget allocation.
    test_video_scores = np.array([-1.0, 1.5, 0.2, 0.8])
    test_allocation = allocate_budget(
        test_video_scores,
        total_budget=20,
        b_min=2,
        b_max=10,
    )

    print("Budget allocation:", test_allocation)
    print("Total allocated frames:", np.sum(test_allocation))

    # Test k-center frame selection with simple synthetic features.
    test_features = np.array([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
    ], dtype=np.float32)

    selected_indices = k_center_select(test_features, k=3)

    print("Selected frame indices:", selected_indices)
    print("Number of selected frames:", len(selected_indices))

    # Final sanity checks.
    assert np.sum(test_allocation) == 20
    assert np.all(test_allocation >= 2)
    assert np.all(test_allocation <= 10)

    assert len(selected_indices) == 3
    assert len(set(selected_indices)) == 3

    print("All Day 3 tests passed!")
