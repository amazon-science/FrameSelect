# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""QAV: Query-Answer-Video aligned frame selection.

Training-free, query-aware selector for multiple-choice questions. Each
answer choice ``a_j`` is appended to the question ``Q`` and encoded, and
every candidate frame is scored by its best alignment over the choices,

    ``s_i = max_j cos(v_i, VL(Q ∥ a_j))``

so frames that support any of the answers rank high. The ``k`` best-scoring
frames are returned in temporal order. Without answer choices the score is
``cos(v_i, VL(Q))``, the paper's question-only variant (QV).

Answer choices travel inside the query string, one per line after the
question, in the ``"A. text"`` / ``"(B) text"`` / ``"C) text"`` form the
evaluators use for MCQ options::

    sel.select(video, query="What is the man holding?\\nA. A cup.\\nB. A phone.")

Evaluators pass the choices this way automatically, because the selector
sets :attr:`uses_answer_choices`.

Reference
---------
Islam et al., "Query-aligned video frame selection for long video
understanding", 2026.
Paper: https://arxiv.org/abs/2609.31668
Code: Not publicly available, to the best of our knowledge

Details the paper leaves open
-----------------------------
* ``Q ∥ a_j`` joins the question and the choice text with a space; the
  choice's letter label is dropped.
* Fewer than two parsed choices means the query is a plain question (QV).
* The paper encodes with CLIP or LongCLIP; here the shared frame encoder
  (CLIP ViT-B-32 by default) is used.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np

from ..base import BaseSelector
from ..registry import register

# "A. text", "a) text", "(B) text", "C: text" at the start of a line.
_CHOICE = re.compile(r"^\s*\(?([A-Za-z])\s*[.):]\s+(\S.*?)\s*$")


def split_query(query: str) -> tuple[str, list[str]]:
    """Split ``"question\\nA. x\\nB. y"`` into the question and the choice texts.

    Lines after the question that start with a letter label are choices; a
    line without a label continues the previous choice. Fewer than two
    choices means there are none: the whole query is the question.
    """
    question, choices = [], []
    for line in query.splitlines():
        if not line.strip():
            continue
        match = _CHOICE.match(line) if question else None
        if match:
            choices.append(match.group(2))
        elif choices:
            choices[-1] = f"{choices[-1]} {line.strip()}"
        else:
            question.append(line.strip())
    if len(choices) < 2:
        return " ".join(line.strip() for line in query.splitlines() if line.strip()), []
    return " ".join(question), choices


@register("qav")
class QAVSelector(BaseSelector):
    """Query-answer-video aligned selection: top-k by best question+choice alignment.

    Parameters
    ----------
    clip_model, clip_pretrained : str
        OpenCLIP backbone used when feeding raw frames. Default
        ``"ViT-B-32"`` / ``"openai"``.
    """

    query_aware = True
    uses_answer_choices = True

    def __init__(self, device: str = "cpu", **kwargs):
        super().__init__(device=device, **kwargs)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            # Route through the base factory so a decoupled multilingual text
            # tower (``text_encoder="mclip:..."``) is honoured.
            self._encoder = self._construct_encoder()
        return self._encoder

    def _text_embeddings(
        self, query: str, query_embedding: Optional[np.ndarray]
    ) -> np.ndarray:
        """``(n_texts, D)`` embeddings of ``Q ∥ a_j`` for every choice, or of ``Q``."""
        question, choices = split_query(query)
        if not choices:
            if query_embedding is None:
                query_embedding = self._get_encoder().encode_text(question)
            return np.atleast_2d(np.asarray(query_embedding, dtype=np.float32))
        encoder = self._get_encoder()
        return np.stack([
            np.asarray(encoder.encode_text(f"{question} {choice}"), dtype=np.float32)
            for choice in choices
        ])

    def _run(self, embeddings: np.ndarray, texts: np.ndarray, n_select: int):
        emb = np.asarray(embeddings, dtype=np.float32)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        texts = texts / (np.linalg.norm(texts, axis=1, keepdims=True) + 1e-12)
        scores = (emb @ texts.T).max(axis=1)
        n_select = min(n_select, len(scores))
        top = np.argsort(-scores, kind="stable")[:n_select]
        indices = np.sort(top).astype(np.int64)
        return indices, scores[indices], scores, "cosine"

    def _select_indices(
        self,
        frames: np.ndarray,
        n_select: int,
        query: Optional[str] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        if query is None:
            raise ValueError("QAV requires a textual query.")
        frame_embs = self._get_encoder().encode_frames(frames).astype(np.float32)
        return self._run(frame_embs, self._text_embeddings(query, None), n_select)

    def _select_from_embeddings(
        self,
        embeddings: np.ndarray,
        n_select: int,
        query: Optional[str] = None,
        query_embedding: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        if query is None:
            raise ValueError("QAV requires a textual query.")
        return self._run(embeddings, self._text_embeddings(query, query_embedding), n_select)
