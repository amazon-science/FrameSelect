# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Uniform / rule-based frame sampling.

One selector covers both common rule-based strategies:

- **Classic uniform**: pick ``n_frames`` evenly spaced indices from the
  candidate pool (default behaviour).
- **FPS sampling**: pick indices at a target FPS stride. Enable via
  ``target_fps``; paired with ``native_fps`` this samples one frame
  every ``native_fps / target_fps`` positions, capped at ``n_frames``.

To the best of our knowledge, FPS sampling is the default strategy in
most open-source video datasets and evaluation harnesses. In
FrameSelect's two-stage contract the caller usually pre-samples
candidates at a given FPS via ``select(..., candidate_fps=...)``, so
pure evenly-spaced selection is sufficient for most workflows. The
``target_fps`` knob exists for people who want the classical
semantics in one place.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..base import BaseSelector
from ..registry import register


def _linspace(T: int, k: int) -> np.ndarray:
    k = min(k, T)
    return np.linspace(0, T - 1, k, dtype=np.int64)


def _fps_stride(T: int, k: int, native_fps: float, target_fps: float) -> np.ndarray:
    """Fixed-FPS sampling with graceful fallback."""
    if native_fps is None or native_fps <= 0 or target_fps is None or target_fps <= 0:
        return _linspace(T, k)
    step = max(1, int(round(native_fps / target_fps)))
    indices = np.arange(0, T, step, dtype=np.int64)
    if len(indices) >= k:
        # Evenly pick k of them to respect the frame budget
        return indices[_linspace(len(indices), k)]
    # Not enough — fall back to uniform to honour ``n_frames``
    return _linspace(T, k)


@register("uniform")
class UniformSelector(BaseSelector):
    """Evenly spaced frame sampling with optional target-FPS mode.

    Parameters
    ----------
    target_fps : float, optional
        If given, sample at a fixed stride determined by the video's
        native FPS (requires ``native_fps``), then evenly
        downsample to ``n_frames``. Leave as ``None`` for classic
        evenly-spaced selection — good default for fixed frame
        budgets. The stride is counted in candidate positions, so it
        means native frames only when every frame is a candidate:
        ``candidate_fps=0`` and ``candidate_frames=0`` (no pool cap).
    native_fps : float, optional
        The source video's FPS for the stride calculation. Only used
        when ``target_fps`` is set; if it is ``None``, the method falls
        back to evenly spaced selection.

    Notes
    -----
    FrameSelect's usual pattern is to pre-sample candidate frames at
    some ``candidate_fps`` via ``select(..., candidate_fps=...)``, so
    classic uniform on the candidate pool already implements "sample
    at X FPS then pick N". The ``target_fps`` parameter here exists
    to also let you do both in one selector call.
    """

    query_aware = False
    # Selection touches no learned components (just np.linspace on
    # decord-decoded frame indices), so running many video selections
    # concurrently is purely I/O-bound on video header reads and
    # scales linearly up to ``n_workers``.
    parallel_selection = True

    def __init__(
        self,
        device: str = "cpu",
        target_fps: Optional[float] = None,
        native_fps: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        self.target_fps = target_fps
        self.native_fps = native_fps

    def _hyperparams(self) -> dict:
        return {"target_fps": self.target_fps, "native_fps": self.native_fps}

    def _select_indices(
        self,
        frames: np.ndarray,
        n_select: int,
        query: Optional[str] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        T = len(frames)
        if self.target_fps is not None and self.native_fps is not None:
            return _fps_stride(T, n_select, self.native_fps, self.target_fps), None
        return _linspace(T, n_select), None

    def _select_from_embeddings(
        self,
        embeddings: np.ndarray,
        n_select: int,
        query: Optional[str] = None,
        query_embedding: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        T = len(embeddings)
        if self.target_fps is not None and self.native_fps is not None:
            return _fps_stride(T, n_select, self.native_fps, self.target_fps), None
        return _linspace(T, n_select), None
