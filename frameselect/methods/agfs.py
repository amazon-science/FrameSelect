# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""AGFS: Adaptive Greedy Frame Selection.

Training-free, query-aware selector that picks the ``K`` frames maximising a
relevance term plus a facility-location coverage term,

    ``F(S) = α · Σ_{i ∈ S} r_i  +  β · (1/N) Σ_{j=1..N} (max(b, max_{i ∈ S} s_{j,i}) − b)``

with ``r_i = max(cos(v_i, t), 0)`` the clipped frame-query similarity,
``s_{j,i}`` the frame-frame similarity and ``b = −1`` (so ``F(∅) = 0``).
``F`` is normalised, monotone and submodular, so the greedy algorithm, which
adds the frame of largest marginal gain until ``K`` are chosen, carries the
standard ``(1 − 1/e)`` guarantee. Coverage rewards a set that represents the
whole candidate pool; relevance rewards frames that match the question.

The paper studies four presets of ``(α, β)``:

* ``"coverage_only"``      — ``(0, 1)``
* ``"coverage_oriented"``  — ``(0.5, 1)``
* ``"relevance_only"``     — ``(1, 0)``
* ``"relevance_oriented"`` — ``(1, 0.5)``

and routes each question to one of them with a text-only question-type
classifier trained on MLVU. To the best of our knowledge, the paper specifies
that classifier no further than its training split, so it is not reproduced
here: pick a
preset (default ``"relevance_oriented"``, the best fixed preset in most of the
paper's settings) or pass ``alpha`` and ``beta`` directly.

Reference
---------
Huang et al., "Adaptive Greedy Frame Selection for Long Video
Understanding", 2026.
Paper: https://arxiv.org/abs/2603.20180
Code: Not publicly available, to the best of our knowledge

Encoder note
------------
The paper scores relevance with SigLIP and measures frame similarity with
DINOv2. For apples-to-apples with the other methods in FrameSelect, both come
from the shared frame encoder (CLIP ViT-B-32 by default; the FrameSelect
paper's shared encoder is ``encoder="ViT-SO400M-16-SigLIP2-384"``). The paper's candidate pool
is every second of the video capped at 1,000; the library's pool settings
(``candidate_fps``, ``candidate_frames``) apply instead.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..base import BaseSelector
from ..registry import register

PRESETS: dict[str, tuple[float, float]] = {
    "coverage_only": (0.0, 1.0),
    "coverage_oriented": (0.5, 1.0),
    "relevance_only": (1.0, 0.0),
    "relevance_oriented": (1.0, 0.5),
}

# Floor of the facility-location coverage: the lowest possible cosine.
_COVERAGE_FLOOR = -1.0


def _greedy_select(
    relevance: np.ndarray,
    similarity: np.ndarray,
    n_select: int,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Greedy maximisation of ``α R(S) + β C(S)``; returns sorted indices.

    ``relevance`` holds the clipped scores ``r_i``; ``similarity`` the
    ``(N, N)`` frame-frame matrix. Ties go to the earliest frame.
    """
    N = len(relevance)
    n_select = min(n_select, N)
    coverage = np.full(N, _COVERAGE_FLOOR, dtype=np.float64)
    available = np.ones(N, dtype=bool)
    picked: list[int] = []
    for _ in range(n_select):
        # Marginal coverage gain of every candidate i: (1/N) Σ_j [max(c_j, s_ji) − c_j].
        gain = (np.maximum(similarity, coverage[:, None]) - coverage[:, None]).sum(axis=0) / N
        score = alpha * relevance + beta * gain
        score[~available] = -np.inf
        best = int(np.argmax(score))
        picked.append(best)
        available[best] = False
        coverage = np.maximum(coverage, similarity[:, best])
    return np.sort(np.asarray(picked, dtype=np.int64))


@register("agfs")
class AGFSSelector(BaseSelector):
    """Adaptive Greedy Frame Selection: relevance + facility-location coverage.

    Parameters
    ----------
    preset : str
        One of ``"coverage_only"``, ``"coverage_oriented"``,
        ``"relevance_only"`` and ``"relevance_oriented"`` (default).
    alpha, beta : float, optional
        Explicit relevance and coverage weights; when either is given both
        must be, and they override ``preset`` (recorded as ``None``). With
        ``alpha = 0`` (coverage only) the instance is query-agnostic.
    clip_model, clip_pretrained : str
        OpenCLIP backbone used when feeding raw frames. Default
        ``"ViT-B-32"`` / ``"openai"``.
    """

    query_aware = True

    def __init__(
        self,
        device: str = "cpu",
        preset: str = "relevance_oriented",
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        if preset not in PRESETS:
            raise ValueError(f"preset must be one of {sorted(PRESETS)}, got {preset!r}")
        if (alpha is None) != (beta is None):
            raise ValueError("pass both alpha and beta, or neither")
        if alpha is not None:
            if alpha < 0 or beta < 0 or alpha + beta == 0:
                raise ValueError(f"alpha and beta must be >= 0 and not both 0, got {alpha}, {beta}")
            weights = (float(alpha), float(beta))
        else:
            weights = PRESETS[preset]
        # Explicit weights override the preset, so provenance records none.
        self.preset = preset if alpha is None else None
        self.alpha, self.beta = weights
        if self.alpha == 0:
            # Coverage only: the picks do not depend on the query.
            self.query_aware = False
        self._encoder = None

    def _hyperparams(self) -> dict:
        return {"preset": self.preset, "alpha": self.alpha, "beta": self.beta}

    def _get_encoder(self):
        if self._encoder is None:
            # Route through the base factory so a decoupled multilingual text
            # tower (``text_encoder="mclip:..."``) is honoured.
            self._encoder = self._construct_encoder()
        return self._encoder

    def _run(self, embeddings: np.ndarray, query_cosine: Optional[np.ndarray], n_select: int):
        emb = np.asarray(embeddings, dtype=np.float32)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        if query_cosine is None:
            relevance = np.zeros(len(emb), dtype=np.float32)
        else:
            relevance = np.maximum(np.asarray(query_cosine, dtype=np.float32), 0.0)
        similarity = (emb @ emb.T).astype(np.float64)
        indices = _greedy_select(relevance.astype(np.float64), similarity, n_select,
                                 self.alpha, self.beta)
        return indices, relevance[indices], relevance, "cosine"

    def _need_query(self, query: Optional[str]) -> None:
        if query is None and self.alpha > 0:
            raise ValueError("AGFS requires a textual query unless alpha is 0 (coverage only).")

    def _select_indices(
        self,
        frames: np.ndarray,
        n_select: int,
        query: Optional[str] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        self._need_query(query)
        encoder = self._get_encoder()
        frame_embs = encoder.encode_frames(frames).astype(np.float32)
        cosine = None
        if query is not None:
            cosine = frame_embs @ encoder.encode_text(query).astype(np.float32)
        return self._run(frame_embs, cosine, n_select)

    def _select_from_embeddings(
        self,
        embeddings: np.ndarray,
        n_select: int,
        query: Optional[str] = None,
        query_embedding: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        self._need_query(query)
        emb = embeddings.astype(np.float32)
        cosine = None
        if query is not None:
            if query_embedding is None:
                query_embedding = self._get_encoder().encode_text(query)
            cosine = emb @ np.asarray(query_embedding, dtype=np.float32)
        return self._run(emb, cosine, n_select)
