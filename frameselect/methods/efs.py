# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""EFS: Event-Anchored Frame Selection.

Training-free, query-driven selector that treats the video as a sequence of
visual events rather than a flat bag of frames. Three stages:

1. **Visual event partitioning.** A temporal-similarity curve

   ``s_i = Σ_{j ∈ N(i)} w_|i−j| · cos(f_i, f_j) / Σ_{j ∈ N(i)} w_|i−j|``

   over a linearly weighted neighbourhood marks event boundaries at its
   local minima (the points of largest visual change). While there are
   more than ``max_events`` events, the adjacent pair whose mean features
   are most similar is merged.
2. **Event anchor localization.** The most query-relevant frame of each
   event becomes its anchor, so the initial set covers every event.
3. **Anchor-guided global refinement** (Alg. 1 in the paper). The other
   frames, in descending relevance, join the set while their highest
   similarity to the frames already chosen stays below a threshold that
   starts at ``μ − α·σ`` and is relaxed in steps of ``delta`` up to
   ``μ + α·σ``, where μ and σ are the mean and standard deviation of every
   non-anchor frame's similarity to its nearest anchor. The threshold thus
   adapts to how visually redundant each video is.

Reference
---------
Chen et al., "Event-Anchored Frame Selection for Effective Long-Video
Understanding", 2026.
Paper: https://arxiv.org/abs/2603.00983
Code: Not publicly available, to the best of our knowledge

Encoder note
------------
The paper scores query relevance with BLIP-2's image-text matching head and
measures frame similarity with DINOv2 features. For apples-to-apples with
the other methods in FrameSelect, both come from the shared frame encoder
(CLIP ViT-B-32 by default): relevance is the frame-query cosine and
similarity the frame-frame cosine. The algorithm itself is encoder-agnostic.

Details the paper leaves open
-----------------------------
* ``window`` (the paper's ``l = 3``) is read as the neighbourhood's
  half-width, with weights ``w_d = window + 1 − d`` for ``d = 1..window``.
* A local minimum of the curve starts a new event; on a flat minimum (a
  cut dips the frames on both sides equally) the later frame does.
* The event count is capped at the frame budget, so every anchor is kept.
* To the best of our knowledge, the paper does not state the increment ``delta``; the default is 0.01.
* If the loosest threshold still leaves the set short of the budget, the
  most relevant remaining frames fill it, so the output always holds
  ``n_select`` distinct frames.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..base import BaseSelector
from ..registry import register


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def _temporal_similarity(emb: np.ndarray, window: int) -> np.ndarray:
    """Weighted mean cosine similarity of each frame to its neighbours.

    ``emb`` must be L2-normalised. Neighbours ``j`` with
    ``1 <= |i - j| <= window`` contribute with weight ``window + 1 - |i - j|``;
    edge frames average over the neighbours they have. A single frame is
    perfectly self-similar (1.0).
    """
    T = len(emb)
    num = np.zeros(T, dtype=np.float64)
    den = np.zeros(T, dtype=np.float64)
    for d in range(1, min(window, T - 1) + 1):
        w = float(window + 1 - d)
        cos = np.einsum("ij,ij->i", emb[:-d], emb[d:])
        num[:-d] += w * cos
        den[:-d] += w
        num[d:] += w * cos
        den[d:] += w
    curve = np.ones(T, dtype=np.float64)
    mask = den > 0
    curve[mask] = num[mask] / den[mask]
    return curve


def _partition_events(
    emb: np.ndarray, curve: np.ndarray, max_events: int
) -> list[tuple[int, int]]:
    """Split ``[0, T)`` at the local minima of ``curve``; merge down to ``max_events``.

    Returns half-open ``(start, end)`` intervals in temporal order. The
    adjacent pair whose L2-normalised mean features have the highest cosine
    similarity is merged first.
    """
    T = len(curve)
    # A cut between frames k-1 and k dips both equally under a symmetric
    # window; taking the last frame of a flat minimum starts the new event
    # at k, the first frame after the change.
    minima = [
        i for i in range(1, T - 1)
        if curve[i] <= curve[i - 1] and curve[i] < curve[i + 1]
    ]
    starts = [0] + minima
    events = list(zip(starts, starts[1:] + [T]))
    if len(events) <= max_events:
        return events

    means = [_l2_normalize(emb[s:e].mean(axis=0)) for s, e in events]
    sims = [float(means[k] @ means[k + 1]) for k in range(len(events) - 1)]
    while len(events) > max_events:
        k = int(np.argmax(sims))
        merged = (events[k][0], events[k + 1][1])
        events[k:k + 2] = [merged]
        means[k:k + 2] = [_l2_normalize(emb[merged[0]:merged[1]].mean(axis=0))]
        del sims[k]
        if k > 0:
            sims[k - 1] = float(means[k - 1] @ means[k])
        if k < len(sims):
            sims[k] = float(means[k] @ means[k + 1])
    return events


def _refine(
    emb: np.ndarray,
    relevance: np.ndarray,
    anchors: np.ndarray,
    n_select: int,
    alpha: float,
    delta: float,
) -> np.ndarray:
    """Anchor-guided global refinement (Alg. 1); returns sorted indices."""
    T = len(relevance)
    selected = [int(a) for a in anchors]
    in_set = np.zeros(T, dtype=bool)
    in_set[selected] = True
    order = np.argsort(-relevance, kind="stable")

    if len(selected) < n_select:
        rest = ~in_set
        nearest_anchor = (emb[rest] @ emb[selected].T).max(axis=1)
        mu, sigma = float(nearest_anchor.mean()), float(nearest_anchor.std())
        strict = float(np.clip(mu - alpha * sigma, 0.0, 1.0))
        loose = float(np.clip(mu + alpha * sigma, 0.0, 1.0))
        # Highest similarity of every frame to the current set, kept up to
        # date as frames join so each test sees the set as it is now.
        max_sim = (emb @ emb[selected].T).max(axis=1)
        threshold = strict
        while len(selected) < n_select:
            for c in order:
                if len(selected) >= n_select:
                    break
                if in_set[c] or max_sim[c] >= threshold:
                    continue
                selected.append(int(c))
                in_set[c] = True
                max_sim = np.maximum(max_sim, emb @ emb[c])
            if threshold >= loose:
                break
            threshold = min(threshold + delta, loose)

    # Fill any shortfall with the most relevant frames not yet chosen.
    for c in order:
        if len(selected) >= n_select:
            break
        if not in_set[c]:
            selected.append(int(c))
            in_set[c] = True
    return np.sort(np.asarray(selected, dtype=np.int64))


def _efs_select(
    emb: np.ndarray,
    relevance: np.ndarray,
    n_select: int,
    *,
    max_events: int,
    alpha: float,
    delta: float,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Full EFS pipeline; returns ``(sorted indices, temporal-similarity curve)``."""
    emb = _l2_normalize(emb)
    relevance = np.asarray(relevance, dtype=np.float32)
    curve = _temporal_similarity(emb, window)
    n_select = min(n_select, len(relevance))
    if n_select <= 0:
        return np.zeros(0, dtype=np.int64), curve
    events = _partition_events(emb, curve, min(max_events, n_select))
    anchors = np.array(
        [s + int(np.argmax(relevance[s:e])) for s, e in events], dtype=np.int64
    )
    indices = _refine(emb, relevance, anchors, n_select, alpha, delta)
    return indices, curve


@register("efs")
class EFSSelector(BaseSelector):
    """Event-Anchored Frame Selection: event anchors + adaptive MMR refinement.

    Parameters
    ----------
    max_events : int
        Target number of visual events ``M``. Paper default ``10`` (best on
        Video-MME; 14 on LongVideoBench and MLVU). Capped at the frame
        budget.
    alpha : float
        Threshold relaxation factor ``α`` of the adaptive refinement. Paper
        default ``0.5`` (best on Video-MME; 0.1 on LongVideoBench and MLVU).
    delta : float
        Threshold increment per refinement round. Not stated in the paper,
        to the best of our knowledge; default ``0.01``.
    window : int
        Half-width ``l`` of the neighbourhood behind the temporal-similarity
        curve. Paper default ``3``.
    clip_model, clip_pretrained : str
        OpenCLIP backbone used when feeding raw frames. Default
        ``"ViT-B-32"`` / ``"openai"``.
    """

    query_aware = True

    def __init__(
        self,
        device: str = "cpu",
        max_events: int = 10,
        alpha: float = 0.5,
        delta: float = 0.01,
        window: int = 3,
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        if max_events < 1:
            raise ValueError(f"max_events must be >= 1, got {max_events}")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if delta <= 0:
            raise ValueError(f"delta must be > 0, got {delta}")
        self.max_events = max_events
        self.alpha = alpha
        self.delta = delta
        self.window = window
        self._encoder = None

    def _hyperparams(self) -> dict:
        return {
            "max_events": self.max_events,
            "alpha": self.alpha,
            "delta": self.delta,
            "window": self.window,
        }

    def _get_encoder(self):
        if self._encoder is None:
            # Route through the base factory so a decoupled multilingual text
            # tower (``text_encoder="mclip:..."``) is honoured.
            self._encoder = self._construct_encoder()
        return self._encoder

    def _run(self, embeddings: np.ndarray, relevance: np.ndarray, n_select: int):
        indices, curve = _efs_select(
            embeddings, relevance, n_select,
            max_events=self.max_events, alpha=self.alpha,
            delta=self.delta, window=self.window,
        )
        self._pending_debug_traces = {
            "temporal_similarity": {
                "values": curve.astype(np.float32),
                "label": "Temporal similarity",
                "kind": "curve",
                "higher_is_better": True,
            },
            "relevance": {
                "values": relevance.astype(np.float32),
                "label": "Query relevance",
                "kind": "curve",
                "higher_is_better": True,
            },
        }
        return indices, relevance[indices], relevance, "cosine"

    def _select_indices(
        self,
        frames: np.ndarray,
        n_select: int,
        query: Optional[str] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        if query is None:
            raise ValueError("EFS requires a textual query.")
        encoder = self._get_encoder()
        frame_embs = encoder.encode_frames(frames).astype(np.float32)
        text_emb = encoder.encode_text(query).astype(np.float32)
        return self._run(frame_embs, frame_embs @ text_emb, n_select)

    def _select_from_embeddings(
        self,
        embeddings: np.ndarray,
        n_select: int,
        query: Optional[str] = None,
        query_embedding: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        if query is None:
            raise ValueError("EFS requires a textual query.")
        if query_embedding is None:
            query_embedding = self._get_encoder().encode_text(query)
        emb = embeddings.astype(np.float32)
        q = np.asarray(query_embedding, dtype=np.float32)
        return self._run(emb, emb @ q, n_select)
