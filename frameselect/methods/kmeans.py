# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""K-Means clustering on CLIP embeddings for diversity-driven selection.

Query-agnostic content-based selector. Runs K-Means with
``k = n_select`` clusters directly on CLIP frame embeddings and
keeps the frame whose embedding is closest to each cluster centroid.
Each cluster captures a distinct visual "theme", so the final set is
forced to be visually diverse.

How it relates to other methods in this library
-----------------------------------------------
- ``uniform`` — the query- and content-agnostic baseline: same budget,
  evenly spaced in time instead of spread over visual themes.
- ``agfs`` — the same coverage idea as a facility-location objective,
  greedily combined with query relevance.
- ``efs`` and ``qav`` — use the same embeddings but also score relevance
  to the query, so they trade some diversity for coverage of what the
  question asks about.

Reference
---------
K-Means on learned visual features is a long-standing baseline in
video summarisation and frame selection.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..base import BaseSelector
from ..registry import register


@register("kmeans")
class KMeansSelector(BaseSelector):
    """K-Means clustering on CLIP frame embeddings.

    Steps:
    1. Extract CLIP embeddings for all candidate frames.
    2. Run K-Means with ``k = n_select`` clusters.
    3. For each cluster, pick the frame nearest to the centroid.

    Parameters
    ----------
    clip_model : str
        OpenCLIP model name. Default ``"ViT-B-32"``.
    clip_pretrained : str
        Pretrained weights tag. Default ``"openai"``.
    """

    query_aware = False

    def __init__(
        self,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from ..encoders import CLIPEncoder

            self._encoder = CLIPEncoder(
                self.clip_model, self.clip_pretrained, self.device
            )
        return self._encoder

    def _select_indices(
        self,
        frames: np.ndarray,
        n_select: int,
        query: Optional[str] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
        from sklearn.cluster import KMeans

        encoder = self._get_encoder()
        embeddings = encoder.encode_frames(frames)  # (T, D)

        kmeans = KMeans(n_clusters=n_select, random_state=self._seed_legacy(42), n_init=10)
        kmeans.fit(embeddings)

        # Per-candidate "representativeness" score: negative distance to
        # the assigned cluster's centroid, so larger = closer to a
        # cluster centre. Selected frames are the argmax within each
        # cluster by construction.
        assigned_centroids = kmeans.cluster_centers_[kmeans.labels_]
        dists = np.linalg.norm(embeddings - assigned_centroids, axis=1)
        all_scores = (-dists).astype(np.float32)

        # For each cluster, find the nearest frame to the centroid.
        # Skip empty clusters (KMeans can return fewer non-empty clusters
        # than requested — argmin over an empty member set would crash),
        # then backfill to n_select below.
        indices = _pick_centroid_nearest(embeddings, kmeans, n_select)
        scores = all_scores[indices]
        # score_kind "custom": the per-frame value is negative distance to
        # the assigned cluster centroid (larger = more central). See this
        # selector's docstring; kept outside the documented cosine/logit/
        # probability enum on purpose.
        return indices, scores, all_scores, "custom"

    def _select_from_embeddings(
        self,
        embeddings: np.ndarray,
        n_select: int,
        query: Optional[str] = None,
        query_embedding: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=n_select, random_state=self._seed_legacy(42), n_init=10)
        kmeans.fit(embeddings)

        assigned_centroids = kmeans.cluster_centers_[kmeans.labels_]
        dists = np.linalg.norm(embeddings - assigned_centroids, axis=1)
        all_scores = (-dists).astype(np.float32)

        indices = _pick_centroid_nearest(embeddings, kmeans, n_select)
        scores = all_scores[indices]
        return indices, scores, all_scores, "custom"


def _pick_centroid_nearest(
    embeddings: np.ndarray, kmeans, n_select: int
) -> np.ndarray:
    """Pick the centroid-nearest frame per cluster, then backfill to n_select.

    KMeans may leave some of the requested ``n_select`` clusters empty
    (fewer distinct points than clusters). Taking ``argmin`` over an
    empty member set raises, so we skip empty clusters and deterministically
    backfill with the remaining unpicked frames (by original index order)
    until we have ``min(n_select, n_frames)`` distinct picks.
    """
    n_frames = len(embeddings)
    picks: list[int] = []
    seen: set[int] = set()
    for c in range(kmeans.cluster_centers_.shape[0]):
        members = np.where(kmeans.labels_ == c)[0]
        if len(members) == 0:
            continue
        centroid = kmeans.cluster_centers_[c]
        member_dists = np.linalg.norm(embeddings[members] - centroid, axis=1)
        idx = int(members[np.argmin(member_dists)])
        if idx not in seen:
            seen.add(idx)
            picks.append(idx)

    n_target = min(n_select, n_frames)
    if len(picks) < n_target:
        for idx in range(n_frames):
            if len(picks) >= n_target:
                break
            if idx not in seen:
                seen.add(idx)
                picks.append(idx)

    return np.sort(np.array(picks, dtype=np.int64))
