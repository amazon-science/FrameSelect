# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Dataset-agnostic evaluator base class.

Each concrete evaluator (e.g. ActivityNet) supplies:
  * an ``Example`` stream via :meth:`load_annotations`
  * optional subgroup aggregation (ActivityNet groups by top-level category)

The rest — selector dispatch (including the replay fast path, the
per-question cache fast path, thread-parallel selection, and the
VLM call fan-out) — is shared.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from ..video_io import _capped_positions
from ._cache_check import (  # noqa: F401  (CacheMismatchError re-exported)
    CacheMismatchError,
    check_cache_matches_selector,
    encoder_output_dim,
    read_cache_meta,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VLM backend resolution — env-var only (no global VLM singleton).
#
# Resolution is **lazy**: it must not happen at import time, because the
# ``localhost:8000`` vLLM probe (priority 2) does network I/O and would
# make ``import frameselect.evaluations.base`` block on a socket connect.
# Instead the backend is resolved on first access to ``_VLM_BACKEND``
# (via the module ``__getattr__`` below) or on the first evaluator
# construction, and cached. Setting ``base._VLM_BACKEND`` explicitly
# shadows the lazy resolver; ``_get_vlm_backend`` honours it.
# ---------------------------------------------------------------------------

# Failure sentinel for the answering VLM. A failed VLM call returns this
# (introduced in ``vlm/_mcq_parse.py``) instead of silently defaulting to
# ``"A"`` — so a benchmark counts the failure as incorrect rather than as
# a spuriously-correct "A". Imported with a local fallback so this
# module stays importable on its own.
try:
    from ..vlm._mcq_parse import FAILED_ANSWER  # type: ignore
except Exception:  # pragma: no cover - forward/backward compat shim
    FAILED_ANSWER = "__FRAMESELECT_VLM_FAILED__"


def _resolve_eval_backend():
    """Return the backend string for the answering VLM.

    Priority:
    1. ``OPENAI_BASE_URL`` env var → openai
    2. Auto-detect: probe ``localhost:8000`` for a running vLLM server
    3. ``HF_VLM_MODEL`` env var → hf
    4. AWS Bedrock (if boto3 + credentials available)
    5. Anthropic (fallback)

    There is no global VLM singleton — evaluators receive the
    answering-VLM spec via ``vlm=`` at construction, and selectors
    receive the selection-VLM spec the same way.

    Called lazily (never at import) — see the module docstring above.
    """
    if os.environ.get("OPENAI_BASE_URL"):
        return "openai"
    # Auto-detect a local vLLM server before falling back to other backends.
    from ..vlm.openai import _probe_local_vllm
    if _probe_local_vllm() is not None:
        return "openai"
    if os.environ.get("HF_VLM_MODEL"):
        return "hf"
    try:
        from ..vlm.bedrock import _get_all_endpoints  # noqa: F401
        return "bedrock"
    except ImportError:
        return "anthropic"


# Cached lazily-resolved backend. Stored under a private name (not
# ``_VLM_BACKEND``) so the public ``_VLM_BACKEND`` name stays routed
# through ``__getattr__`` unless a test explicitly sets it.
_VLM_BACKEND_CACHE: Optional[str] = None


def _get_vlm_backend() -> str:
    """Return the resolved answering-VLM backend, resolving lazily once.

    Honours an explicitly-set module attribute ``_VLM_BACKEND`` (e.g. a
    test ``monkeypatch.setattr``); otherwise resolves via
    :func:`_resolve_eval_backend` and caches the result.
    """
    explicit = globals().get("_VLM_BACKEND", None)
    if explicit is not None:
        return explicit
    global _VLM_BACKEND_CACHE
    if _VLM_BACKEND_CACHE is None:
        _VLM_BACKEND_CACHE = _resolve_eval_backend()
    return _VLM_BACKEND_CACHE


# Answering-VLM backends the evaluator can dispatch to (see
# ``_get_backend_symbols`` and ``BaseEvaluator._run_vlm``).
_KNOWN_VLM_BACKENDS = ("openai", "hf", "bedrock", "anthropic")

# A ``backend:`` prefix is a bare word before the first ``:``. Model IDs
# that themselves contain ``:`` (Bedrock's ``...-v1:0``) have a ``-`` or
# ``.`` before it, so they read as prefix-less IDs rather than as a backend.
_BACKEND_PREFIX_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*):")


def _split_vlm_spec(vlm_spec: str) -> tuple[Optional[str], str]:
    """Split ``"backend:model"`` into ``(backend, model)``.

    Returns ``(None, spec)`` when the spec carries no ``backend:`` prefix;
    the caller then uses the auto-detected backend. An explicit prefix that
    names no known backend raises ``ValueError``: a typo such as
    ``"bedrok:..."`` must not silently route the run to another provider.
    """
    s = vlm_spec.strip()
    m = _BACKEND_PREFIX_RE.match(s)
    if m is None:
        return None, s
    backend = m.group(1).lower()
    if backend not in _KNOWN_VLM_BACKENDS:
        raise ValueError(
            f"Unknown VLM backend {m.group(1)!r} in vlm={vlm_spec!r}; expected "
            f"one of {', '.join(_KNOWN_VLM_BACKENDS)} (e.g. "
            "'openai:Qwen/Qwen2.5-VL-7B-Instruct'). If the model ID itself contains "
            f"':', prefix it with its backend, e.g. 'openai:{s}'."
        )
    return backend, s[m.end():]


def _get_backend_symbols(backend: Optional[str] = None):
    """Return ``(HAIKU_MODEL_ID, build_mcq_message, call_fn)`` for a backend.

    Lazily imports the per-backend callables. When ``backend`` is None the
    lazily-resolved default backend is used. The ``call_fn`` is
    ``call_haiku_parallel`` for bedrock and ``call_vlm_row``/
    ``call_haiku_row`` otherwise. An unknown backend name raises
    ``ValueError`` rather than falling through to Anthropic.
    """
    backend = backend or _get_vlm_backend()
    if backend == "openai":
        from ..vlm.openai import HAIKU_MODEL_ID, build_mcq_message, call_vlm_row
        return HAIKU_MODEL_ID, build_mcq_message, call_vlm_row
    if backend == "hf":
        from ..vlm.hf import HAIKU_MODEL_ID, build_mcq_message, call_vlm_row
        return HAIKU_MODEL_ID, build_mcq_message, call_vlm_row
    if backend == "bedrock":
        from ..vlm.bedrock import (
            HAIKU_MODEL_ID,
            build_mcq_message,
            call_haiku_parallel,
        )
        return HAIKU_MODEL_ID, build_mcq_message, call_haiku_parallel
    if backend == "anthropic":
        from ..vlm.anthropic import HAIKU_MODEL_ID, build_mcq_message, call_haiku_row
        return HAIKU_MODEL_ID, build_mcq_message, call_haiku_row
    raise ValueError(
        f"Unknown VLM backend {backend!r}; expected one of "
        f"{', '.join(_KNOWN_VLM_BACKENDS)}."
    )


def _get_haiku_model_id() -> str:
    """Default answering-VLM model ID for the resolved backend."""
    return _get_backend_symbols()[0]


def __getattr__(name):
    """PEP 562 module ``__getattr__`` — lazy resolution of backend globals.

    Keeps ``base._VLM_BACKEND`` and ``from .base import _VLM_BACKEND``
    working without doing any network I/O at import time. Also lazily
    exposes the per-backend symbols.
    """
    if name == "_VLM_BACKEND":
        return _get_vlm_backend()
    if name == "HAIKU_MODEL_ID":
        return _get_backend_symbols()[0]
    if name == "build_mcq_message":
        return _get_backend_symbols()[1]
    if name in ("call_haiku_row", "call_haiku_parallel"):
        return _get_backend_symbols()[2]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _normalize_model_id(model_id: Optional[str]) -> str:
    backend = _get_vlm_backend()
    if model_id is None:
        return _get_haiku_model_id()
    if backend == "openai":
        return model_id
    looks_bedrock = model_id.startswith("us.anthropic.") or model_id.endswith("-v1:0")
    if backend == "anthropic" and looks_bedrock:
        n = model_id
        if n.startswith("us.anthropic."):
            n = n[len("us.anthropic."):]
        if n.endswith("-v1:0"):
            n = n[: -len("-v1:0")]
        return n
    if backend == "bedrock" and not looks_bedrock:
        return f"us.anthropic.{model_id}-v1:0"
    return model_id


def _normalize_model_id_for_backend(model_id: Optional[str], backend: str) -> str:
    """Like ``_normalize_model_id`` but takes an explicit backend."""
    if model_id is None:
        return _get_backend_symbols(backend)[0]
    if backend == "openai" or backend == "hf":
        return model_id
    looks_bedrock = model_id.startswith("us.anthropic.") or model_id.endswith("-v1:0")
    if backend == "anthropic" and looks_bedrock:
        n = model_id
        if n.startswith("us.anthropic."):
            n = n[len("us.anthropic."):]
        if n.endswith("-v1:0"):
            n = n[: -len("-v1:0")]
        return n
    if backend == "bedrock" and not looks_bedrock:
        return f"us.anthropic.{model_id}-v1:0"
    return model_id


def _parse_answering_vlm_model(vlm_spec: str) -> str:
    """Strip any ``backend:`` prefix from a VLM spec, returning the bare model ID.

    The evaluator's answering-VLM plumbing still runs through
    :func:`_normalize_model_id` + the active ``_VLM_BACKEND``, so here
    we just pull the raw model identifier out of a ``"backend:model"``
    string. A prefix-less spec is returned as-is; an unknown prefix raises
    ``ValueError`` (see :func:`_split_vlm_spec`).
    """
    if vlm_spec is None:
        return None
    return _split_vlm_spec(vlm_spec)[1]


def _drop_frames_column(df, *, context: str = ""):
    """Drop the bulky ``frames`` column, failing loudly when it is absent.

    The column is only missing when *no* row ever carried frames -- that is,
    when every single selection failed. Dropping with ``errors="ignore"`` would
    let the run continue and save a result scored over nothing. Fail instead,
    and say why.
    """
    if "frames" in df.columns:
        return df.drop(columns=["frames"])
    raise RuntimeError(
        "No frames were selected for any item"
        + (f" ({context})" if context else "")
        + " -- every selection failed, so there is nothing to score. See the "
        "selector's own errors above (for example a model that failed to load, "
        "or CUDA OOM during selection)."
    )


# ---------------------------------------------------------------------------
# Dataset-example dataclass
# ---------------------------------------------------------------------------

@dataclass
class Example:
    """A single (video, question, options, answer) record.

    ``duration`` lets datasets carry a subgroup label (e.g. a duration bucket)
    for aggregation. Free-form ``metadata`` is returned alongside the
    raw per-question record.
    """

    video_path: str
    question: str
    options: list[str]
    answer: str
    question_id: str = ""
    duration: str = ""   # subgroup label (ActivityNet: top-level category), or ""
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluation result container
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """Output of a benchmark run."""

    accuracy: float
    correct: int
    total: int
    n_frames: int
    method: str
    model_id: str
    vlm_backend: str
    latency: dict
    subgroup_accuracy: dict
    df: pd.DataFrame
    elapsed_sec: Optional[float] = None
    # Number of examples whose selection failed / was missing. These are
    # counted as incorrect in ``accuracy`` (``total`` spans every example,
    # not just the scored ones), so this surfaces how much of the score
    # was driven by selection failures vs. genuine wrong answers.
    n_errors: int = 0
    # Configuration that produced the run (selector config, seed,
    # hyperparams, answering VLM, ``image_size``, library version), as built
    # by :meth:`BaseEvaluator._run_provenance`. Saved under ``"provenance"``
    # so a result JSON documents itself; empty for hand-built results.
    provenance: dict = field(default_factory=dict)

    def save(self, path: "str | Path") -> Path:
        """Save the result to a JSON file.

        The top-level keys and the ``per_video`` record schema are stable
        (replay reads them). ``selection_metrics`` and ``provenance`` are
        added when present.

        Parameters
        ----------
        path : str or Path
            Either a directory (file named ``<method>.json`` is created
            inside) or a full file path.

        Returns
        -------
        Path
            The path the file was written to.

        Examples
        --------
        >>> result.save("benchmarks/activitynet/")
        PosixPath('benchmarks/activitynet/uniform.json')

        >>> result.save("benchmarks/activitynet/uniform.json")
        PosixPath('benchmarks/activitynet/uniform.json')
        """
        path = Path(path)
        if path.is_dir() or not path.suffix:
            path.mkdir(parents=True, exist_ok=True)
            fname = f"{self.method}.json"
            path = path / fname
        else:
            path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "accuracy": self.accuracy,
            "correct": self.correct,
            "total": self.total,
            "n_frames": self.n_frames,
            "method": self.method,
            "model_id": self.model_id,
            "vlm_backend": self.vlm_backend,
            "latency": self.latency,
            "elapsed_sec": self.elapsed_sec,
            "n_errors": self.n_errors,
            "subgroup_accuracy": self.subgroup_accuracy,
            "per_video": [
                {
                    "video": row.get("video", ""),
                    "question_id": row.get("question_id", ""),
                    "gt": row.get("gt", ""),
                    "pred": row.get("pred", ""),
                    "correct": bool(row.get("correct", False)),
                    "duration": row.get("duration", ""),
                    "selected_indices": row.get("selected_indices", []),
                    "selection_latency_ms": row.get("selection_latency_ms"),
                }
                for _, row in self.df.iterrows()
            ],
        }
        # An evaluator may attach a metrics dict to the result; persist it so
        # a saved run is self-describing.
        sel_metrics = getattr(self, "selection_metrics", None)
        if sel_metrics:
            data["selection_metrics"] = sel_metrics
        if self.provenance:
            data["provenance"] = self.provenance

        path.write_text(json.dumps(data, indent=2, default=str))
        return path

    def __str__(self) -> str:
        lines = [
            f"  method:   {self.method}",
            f"  accuracy: {self.accuracy:.2f}%  ({self.correct}/{self.total})",
            f"  n_frames: {self.n_frames}",
            f"  backend:  {self.vlm_backend}  ({self.model_id})",
        ]
        if self.subgroup_accuracy:
            sg = "  ".join(f"{k}={v:.1f}%" for k, v in self.subgroup_accuracy.items())
            lines.append(f"  subgroups: {sg}")
        if self.elapsed_sec:
            lines.append(f"  elapsed:  {self.elapsed_sec:.0f}s")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Frame-loading helpers (mirror the embedding-cache frame format)
# ---------------------------------------------------------------------------

def _cache_npz_path(cache_dir: Path, video_path: str) -> Path:
    """The evaluator's per-video cache file: ``<cache_dir>/<video stem>.npz``.

    Swaps whatever video extension the file has (``.mp4``, ``.mkv``, ...),
    as :func:`_precompute_worker` does when it names the files it writes.
    """
    return Path(cache_dir) / Path(video_path).with_suffix(".npz")


def _load_cache(cache_path: str) -> dict:
    """Load precomputed embeddings from disk (frames loaded lazily).

    ``meta`` holds whatever the file records about how it was built
    (encoder, fps, ...; see :func:`read_cache_meta`), which
    :func:`check_cache_matches_selector` compares with the selector.
    """
    npz = np.load(cache_path)
    return {
        "candidate_indices": npz["candidate_indices"],
        "embeddings": npz["embeddings"],
        "n_candidates": int(npz["n_candidates"]),
        "_jpeg_path": cache_path.replace(".npz", "_frames.bin"),
        "meta": read_cache_meta(npz, cache_path),
    }


def _load_frames_by_indices(jpeg_path: str, indices: list[int]) -> np.ndarray:
    """Load specific frames from a binary JPEG cache.

    Supports two on-disk layouts:
    * **Indexed**: 4-byte frame count, then ``n`` × 8-byte offsets, then
      ``(len u32, jpg bytes)*`` records — random access.
    * **Sequential**: 4-byte frame count, then ``(len u32, jpg)*``
      records — read by a sequential scan.
    """
    frames = []
    with open(jpeg_path, "rb") as f:
        n = int.from_bytes(f.read(4), "little")

        first_8 = f.read(8)
        first_offset = int.from_bytes(first_8, "little")
        expected_data_start = 4 + n * 8
        if expected_data_start <= first_offset < expected_data_start + 1_000_000:
            # New format — random access
            for idx in indices:
                if idx < 0 or idx >= n:
                    continue
                f.seek(4 + idx * 8)
                offset = int.from_bytes(f.read(8), "little")
                f.seek(offset)
                length = int.from_bytes(f.read(4), "little")
                frames.append(np.array(Image.open(io.BytesIO(f.read(length)))))
        else:
            # Old format — sequential scan
            f.seek(4)
            all_frames = []
            for _ in range(n):
                length = int.from_bytes(f.read(4), "little")
                all_frames.append(np.array(Image.open(io.BytesIO(f.read(length)))))
            for idx in indices:
                if 0 <= idx < len(all_frames):
                    frames.append(all_frames[idx])
    return np.stack(frames) if frames else np.empty((0,))


def _load_all_frames(jpeg_path: str) -> np.ndarray:
    with open(jpeg_path, "rb") as f:
        n = int.from_bytes(f.read(4), "little")
        first_8 = f.read(8)
        first_offset = int.from_bytes(first_8, "little")
        expected_data_start = 4 + n * 8
        if expected_data_start <= first_offset < expected_data_start + 1_000_000:
            f.seek(first_offset)
        else:
            f.seek(4)
        frames = []
        for _ in range(n):
            length = int.from_bytes(f.read(4), "little")
            frames.append(np.array(Image.open(io.BytesIO(f.read(length)))))
    return np.stack(frames)


def _load_frames_by_abs_indices(cache: dict, abs_indices):
    """Given absolute video frame indices, look them up in the
    cache's candidate list and load the corresponding JPEGs."""
    candidate_indices = cache["candidate_indices"]
    abs_to_local = {int(a): i for i, a in enumerate(candidate_indices)}
    local = [abs_to_local[int(a)] for a in abs_indices if int(a) in abs_to_local]
    frames = _load_frames_by_indices(cache["_jpeg_path"], local)
    abs_kept = np.asarray(
        [int(a) for a in abs_indices if int(a) in abs_to_local], dtype=np.int64
    )
    return abs_kept, frames


def _load_saved_selections(path: str) -> dict:
    """Load ``(video, question_id) → [abs_indices]`` from a prior run.

    Accepts two on-disk shapes: an :class:`EvaluationResult`-style
    dict with a ``per_video`` key, or a bare list of per-question
    records.
    """
    with open(path, "r") as f:
        data = json.load(f)
    # Accept both the wrapped shape ({"per_video": [...]}) and a bare list.
    if isinstance(data, dict):
        records = data.get("per_video", [])
    else:
        records = data
    mapping = {}
    for rec in records:
        key = (rec["video"], str(rec.get("question_id", "")))
        mapping[key] = list(rec.get("selected_indices", []))
    return mapping


class FrameLoadError(RuntimeError):
    """Raised when frames cannot be loaded from either a JPEG sidecar or
    the source video, and the caller asked for strict loading.

    Under ``strict=True`` (the evaluator's ``run()`` path), this replaces
    the silent black-frame fallback: a missing sidecar AND missing source
    video is a real failure that must be counted as an error (so the
    example scores as incorrect) rather than answered on ``np.zeros``
    frames that look valid.
    """


def _select_from_cache(cache, selector, n_frames, query, query_embedding=None,
                      video_path=None, strict=False, score_sink=None,
                      sink_key=None):
    """Select frames using cached embeddings.

    When the JPEG sidecar is missing, falls back to decoding the
    selected frames from the original video via decord. When neither the
    sidecar nor the source video exists, the behaviour depends on
    ``strict``:

    * ``strict=False`` (default) — return zero-filled placeholder frames
      (kept for direct callers).
    * ``strict=True`` — raise :class:`FrameLoadError` so the evaluator's
      ``run()`` loop counts the example as a selection error (and hence
      scores it incorrect) instead of answering on black frames.

    ``score_sink`` is an optional dict. When given, the selector's **dense
    per-candidate score curve** is recorded as
    ``score_sink[sink_key] = (all_scores, candidate_indices)``. Callers
    that only need frames omit it.
    """
    embeddings = cache["embeddings"]
    jpeg_path = cache["_jpeg_path"]
    candidate_indices = cache["candidate_indices"]
    n_candidates = len(embeddings)
    n_select = min(n_frames, n_candidates)
    has_sidecar = os.path.exists(jpeg_path)
    _has_video = bool(video_path and os.path.exists(video_path))

    if strict and not has_sidecar and not _has_video:
        raise FrameLoadError(
            f"no JPEG sidecar ({jpeg_path}) and no source video "
            f"({video_path}) to load frames from"
        )

    def _load_selected_frames(local_indices, abs_indices):
        if has_sidecar:
            return _load_frames_by_indices(jpeg_path, local_indices.tolist())
        if _has_video:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            return vr.get_batch(abs_indices.tolist()).asnumpy()
        return np.zeros((len(abs_indices), 64, 64, 3), dtype=np.uint8)

    def _load_all():
        if has_sidecar:
            return _load_all_frames(jpeg_path)
        if _has_video:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            return vr.get_batch(candidate_indices.tolist()).asnumpy()
        return np.zeros((n_candidates, 64, 64, 3), dtype=np.uint8)

    def _record(res):
        """Stash the dense per-candidate curve (element 2 of the tuple)."""
        if score_sink is None or res is None or len(res) < 3:
            return
        all_scores = res[2]
        if all_scores is None:
            return
        score_sink[sink_key] = (
            np.asarray(all_scores, dtype=np.float64), candidate_indices,
        )

    result = selector._select_from_embeddings(
        embeddings, n_select, query, query_embedding=query_embedding
    )
    if result is not None:
        _record(result)
        local_indices = result[0]
        abs_indices = candidate_indices[local_indices]
        frames = _load_selected_frames(local_indices, abs_indices)
        return abs_indices, frames

    all_frames = _load_all()
    result = selector._select_indices(all_frames, n_select, query)
    _record(result)
    local_indices = result[0]
    abs_indices = candidate_indices[local_indices]
    return abs_indices, all_frames[local_indices]


def _uses_pool_length_only(selector) -> bool:
    """True for selectors whose picks depend only on the candidate count.

    Such a selector reads no pixel values, so :func:`_select_from_video` can
    skip decoding the whole candidate pool for it. A selector opts in with
    the class attribute ``pool_length_only = True``; until it declares the
    flag itself, the built-in ``uniform`` selector is recognised by exact
    type (a subclass may read pixels, so it is not assumed length-only).
    """
    flag = getattr(selector, "pool_length_only", None)
    if flag is not None:
        return bool(flag)
    from ..methods.uniform import UniformSelector
    return type(selector) is UniformSelector


def _select_from_video(video_path, selector, n_frames, query, candidate_fps,
                       max_candidates=128, score_sink=None, sink_key=None):
    """Decode a video, sub-sample a candidate pool, and select frames.

    Every selector is handed the decoded candidate frames -- the same pool
    the cached path loads from the JPEG sidecar -- except pool-length-only
    ones (see :func:`_uses_pool_length_only`), which get shape-only
    placeholders so the whole pool is not decoded for nothing.

    ``score_sink``/``sink_key`` behave exactly as in :func:`_select_from_cache`:
    when supplied, the dense per-candidate score curve is recorded.
    """
    from decord import VideoReader, cpu

    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    native_fps = float(vr.get_avg_fps())
    step = max(1, int(round(native_fps / candidate_fps)))
    candidate_indices = np.arange(0, total, step)
    # Cap the candidate pool at max_candidates, uniformly sampled across
    # the full video, with the rule the cache precompute uses, so results
    # are consistent whether the cache exists or not.
    candidate_indices = candidate_indices[
        _capped_positions(len(candidate_indices), max_candidates)]

    def _record(res):
        if score_sink is None or res is None or len(res) < 3:
            return
        if res[2] is None:
            return
        score_sink[sink_key] = (
            np.asarray(res[2], dtype=np.float64), candidate_indices,
        )

    if not selector.query_aware:
        n_select = min(n_frames, len(candidate_indices))
        if _uses_pool_length_only(selector):
            candidate_frames = np.zeros(
                (len(candidate_indices), 1, 1, 3), dtype=np.uint8
            )
        else:
            # Pixel- / feature-based selectors (kmeans, user selectors)
            # must see the real frames: handing
            # them placeholders made them select from garbage.
            candidate_frames = vr.get_batch(candidate_indices.tolist()).asnumpy()
        result = selector._select_indices(candidate_frames, n_select, None)
        _record(result)
        abs_indices = candidate_indices[result[0]]
        return abs_indices, vr.get_batch(abs_indices.tolist()).asnumpy()

    candidate_frames = vr.get_batch(candidate_indices.tolist()).asnumpy()
    n_select = min(n_frames, len(candidate_indices))
    result = selector._select_indices(candidate_frames, n_select, query)
    _record(result)
    abs_indices = candidate_indices[result[0]]
    return abs_indices, vr.get_batch(abs_indices.tolist()).asnumpy()


def _auto_precompute_cache(
    video_path: str,
    cache_path: str,
    encoder,
    candidate_fps: float,
    max_candidates: int = 128,
) -> None:
    """Encode a video's candidate frames and write the embedding cache.

    ``encoder`` must be a pre-built :class:`~frameselect.encoders.CLIPEncoder`
    (or compatible) shared across all parallel workers. Creating one
    encoder per worker would OOM the GPU.
    """
    try:
        from decord import VideoReader, cpu
        from ..cache import EmbeddingCache

        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        native_fps = float(vr.get_avg_fps())
        step = max(1, int(round(native_fps / candidate_fps)))
        candidate_indices = np.arange(0, total, step)
        # Uniformly sample across the full video, not the first N: taking
        # [:max_candidates] would only see the first
        # max_candidates/candidate_fps seconds of a long video.
        candidate_indices = candidate_indices[
            _capped_positions(len(candidate_indices), max_candidates)]

        candidate_frames = vr.get_batch(candidate_indices.tolist()).asnumpy()
        embeddings = encoder.encode_frames(candidate_frames)

        cache_dir = os.path.dirname(cache_path)
        os.makedirs(cache_dir, exist_ok=True)
        cache = EmbeddingCache(root=cache_dir, save_frames=True)
        key = os.path.basename(cache_path).replace(".npz", "")
        cache.put(
            key,
            candidate_indices=candidate_indices,
            embeddings=embeddings,
            frames=candidate_frames,
            meta={
                "encoder": getattr(encoder, "model_name", "unknown"),
                "source": video_path,
                "native_fps": native_fps,
                "candidate_fps": candidate_fps,
            },
        )
    except Exception as exc:
        # best-effort — don't break the eval, but leave a breadcrumb.
        logger.warning(
            "auto-precompute failed for %s: %s: %s",
            video_path, type(exc).__name__, exc,
        )


def _precompute_worker(args):
    """Top-level worker function — called directly (not via multiprocessing).

    Encodes a shard of videos on one GPU and writes the embedding cache.
    Launched as a subprocess by ``precompute_embeddings_parallel`` with
    ``CUDA_VISIBLE_DEVICES`` set in the environment before any CUDA init.
    """
    video_paths, cache_dir, encoder_name, encoder_pretrained, \
        gpu_id, candidate_fps, max_candidates = args

    import os as _os
    _os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Ensure long-video EOF handling works in worker processes too
    _os.environ.setdefault("DECORD_EOF_RETRY_MAX", "40960")

    from ..encoders import CLIPEncoder
    from ..cache import EmbeddingCache
    from decord import VideoReader, cpu as dcpu

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = CLIPEncoder(encoder_name, encoder_pretrained, device)
    cache = EmbeddingCache(root=cache_dir, save_frames=True)

    done = 0
    position = int(gpu_id) if str(gpu_id).isdigit() else 0
    for video_path in tqdm(video_paths, desc=f"GPU {gpu_id}", position=position, leave=True):
        # Strip any video extension (.mp4, .mkv, ...), like _cache_npz_path.
        stem = Path(video_path).stem
        if cache.has(stem):
            done += 1
            continue
        try:
            vr = VideoReader(video_path, ctx=dcpu(0))
            total = len(vr)
            native_fps = float(vr.get_avg_fps())
            step = max(1, int(round(native_fps / candidate_fps)))
            cand_idx = np.arange(0, total, step)
            # Uniformly sample across the full video, not the first N.
            cand_idx = cand_idx[_capped_positions(len(cand_idx), max_candidates)]
            frames = vr.get_batch(cand_idx.tolist()).asnumpy()
            embeddings = encoder.encode_frames(frames)
            cache.put(
                stem,
                candidate_indices=cand_idx,
                embeddings=embeddings,
                frames=frames,
                meta={
                    "encoder": encoder_name,
                    "pretrained": encoder_pretrained,
                    "source": video_path,
                    "native_fps": native_fps,
                    "candidate_fps": candidate_fps,
                },
            )
            done += 1
        except Exception as e:
            print(f"  [GPU {gpu_id}] ERR {os.path.basename(video_path)}: {type(e).__name__}: {e}")

    # Explicitly free the encoder before exit so nvidia-smi reports a
    # clean GPU even in the tiny window between return and subprocess
    # reap. Also helps when this worker is reused (unusual) without
    # the parent re-spawning.
    try:
        encoder.release()
    except Exception:
        pass
    return done


def _visible_gpu_tokens() -> list[str]:
    """Physical GPU ids this process may use, in its own index order.

    The entries of ``CUDA_VISIBLE_DEVICES`` when it is set, otherwise
    ``0 .. torch.cuda.device_count() - 1`` (empty without CUDA).
    """
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        return [t.strip() for t in env.split(",") if t.strip()]
    try:
        import torch
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        n = 0
    return [str(i) for i in range(n)]


def precompute_embeddings_parallel(
    video_dir: "str | Path",
    cache_dir: "str | Path",
    encoder_name: str = "ViT-B-32",
    encoder_pretrained: str = "openai",
    candidate_fps: float = 1.0,
    max_candidates: int = 128,
    n_gpus: Optional[int] = None,
    gpu_ids: Optional[list[int]] = None,
) -> int:
    """Precompute SigLIP2/CLIP embeddings for all videos in parallel.

    Spawns one process per GPU, each encoding a shard of the videos.
    Uses ``spawn`` start method so each process gets a clean CUDA context
    (``fork`` + CUDA = deadlocks on Linux).

    Parameters
    ----------
    video_dir : str or Path
        Directory containing MP4 files.
    cache_dir : str or Path
        Where to write ``.npz`` + ``_frames.bin`` cache files.
    encoder_name : str
        OpenCLIP model name (e.g. ``"ViT-SO400M-16-SigLIP2-384"``).
    encoder_pretrained : str
        Pretrained weights tag (e.g. ``"webli"``).
    candidate_fps : float
        Sampling rate for candidate frames. Default 1.0.
    max_candidates : int
        Maximum candidates per video. Default 128.
    n_gpus : int, optional
        Number of GPUs to use. Defaults to every visible GPU (one CPU
        worker when there is none).
    gpu_ids : list[int], optional
        GPU indices as this process numbers them, i.e. positions in
        ``CUDA_VISIBLE_DEVICES`` when it is set. Defaults to
        ``list(range(n_gpus))``.

    Returns
    -------
    int
        Total number of videos encoded.
    """
    import multiprocessing as mp

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    visible = _visible_gpu_tokens()
    if gpu_ids is None:
        gpu_ids = list(range(n_gpus if n_gpus is not None else len(visible)))
    # Each worker gets the physical device behind its index, so a parent
    # restricted with CUDA_VISIBLE_DEVICES keeps its children on those GPUs.
    # An index with no visible GPU behind it gets none (a CPU worker), and
    # with no GPU at all one CPU worker encodes everything.
    gpu_ids = [visible[g] if 0 <= g < len(visible) else "" for g in gpu_ids] or [""]
    n_gpus = len(gpu_ids)

    # Accept either a directory path (scan all .mp4) or a list of video paths.
    from ..cache import EmbeddingCache
    existing_cache = EmbeddingCache(root=cache_dir, save_frames=False)

    if isinstance(video_dir, (list, tuple)):
        all_videos = [Path(v) for v in video_dir]
    else:
        all_videos = sorted(Path(video_dir).glob("*.mp4"))

    pending = [
        str(v) for v in all_videos
        if not existing_cache.has(v.stem)
    ]

    # Use subprocess.Popen to launch one Python process per GPU.
    # Each subprocess starts fresh with CUDA_VISIBLE_DEVICES set in its
    # environment — no CUDA re-initialization issues.
    if not pending:
        print(f"  All {len(all_videos)} videos already cached.")
        return 0

    print(f"  Precomputing embeddings for {len(pending)} videos "
          f"across {n_gpus} GPUs ({encoder_name})...")

    # Split into shards
    shards = [[] for _ in range(n_gpus)]
    for i, v in enumerate(pending):
        shards[i % n_gpus].append(v)

    # Write one JSON config file per shard. The config carries ALL
    # parameters (video paths, cache dir, encoder, fps, ...) so we never
    # interpolate variable values into Python source — the subprocess
    # bootstrap below is a fixed constant that only reads ``sys.argv[1]``
    # (the config path we control). This closes the code-injection /
    # quoting-fragility hole of the old f-string-built ``python -c`` body.
    import json as _json
    import subprocess
    import sys

    repo_root = str(Path(__file__).resolve().parent.parent.parent)

    shard_files = []
    for i, shard in enumerate(shards):
        if not shard:
            continue
        sf = str(cache_dir / f"_shard_{i}.json")
        with open(sf, "w") as f:
            _json.dump(
                {
                    "repo_root": repo_root,
                    "video_paths": shard,
                    "cache_dir": str(cache_dir),
                    "encoder_name": encoder_name,
                    "encoder_pretrained": encoder_pretrained,
                    "gpu_id": gpu_ids[i],
                    "candidate_fps": candidate_fps,
                    "max_candidates": max_candidates,
                },
                f,
            )
        shard_files.append((i, gpu_ids[i], sf))

    # Fixed (non-interpolated) subprocess bootstrap: reads the config
    # path from argv, loads params from JSON, and runs the worker.
    _BOOTSTRAP = (
        "import sys, os, json\n"
        "cfg = json.load(open(sys.argv[1]))\n"
        "sys.path.insert(0, cfg['repo_root'])\n"
        "os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg['gpu_id'])\n"
        "os.environ.setdefault('DECORD_EOF_RETRY_MAX', '40960')\n"
        "from frameselect.evaluations.base import _precompute_worker\n"
        "n = _precompute_worker((cfg['video_paths'], cfg['cache_dir'], "
        "cfg['encoder_name'], cfg['encoder_pretrained'], cfg['gpu_id'], "
        "cfg['candidate_fps'], cfg['max_candidates']))\n"
        "print('[GPU %s] encoded %s videos' % (cfg['gpu_id'], n))\n"
    )

    # Launch one subprocess per GPU
    procs = []
    for shard_id, gpu_id, sf in shard_files:
        cmd = [sys.executable, "-c", _BOOTSTRAP, sf]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env.setdefault("DECORD_EOF_RETRY_MAX", "40960")
        p = subprocess.Popen(cmd, env=env)
        procs.append(p)

    # Wait for all subprocesses
    for p in procs:
        p.wait()

    # Cleanup shard files
    for _, _, sf in shard_files:
        try:
            os.unlink(sf)
        except OSError:
            pass

    # Count results
    total = sum(
        1 for v in all_videos
        if existing_cache.has(Path(v).stem)
    )
    print(f"  Done: {total} videos encoded.")
    return total


# Single-file query-embedding cache that records no encoder, so run() only
# trusts it for the default ViT-B-32.
_LEGACY_QUERY_EMBEDDINGS = "query_embeddings.npz"


def _query_embeddings_filename(encoder_name: str, encoder_pretrained) -> str:
    """Per-encoder query-embedding file inside an evaluator cache_dir."""
    def _safe(s):
        return str(s).replace("/", "_")
    return f"query_embeddings__{_safe(encoder_name)}__{_safe(encoder_pretrained)}.npz"


def _read_query_embeddings(path: "str | Path") -> dict:
    """``{question_id: embedding}`` from a query-embedding cache file."""
    # Query-embedding caches store only a string array of question IDs +
    # a numeric embedding matrix — no pickled Python objects — so
    # ``allow_pickle`` is unnecessary (and disabling it avoids the
    # arbitrary-code-execution surface of pickled .npz files).
    data = np.load(str(path))
    return {str(q): e for q, e in zip(data["question_ids"], data["embeddings"])}


def _reads_embeddings(selector) -> bool:
    """True when ``selector`` overrides ``_select_from_embeddings``, i.e. its
    cached-path selection can read the cached (and query) embeddings.
    Selectors without one pick from the JPEG frames."""
    from ..base import BaseSelector

    impl = getattr(type(selector), "_select_from_embeddings", None)
    return impl is not None and impl is not BaseSelector._select_from_embeddings


def precompute_query_embeddings(
    examples: list,
    cache_dir: "str | Path",
    encoder_name: str = "ViT-B-32",
    encoder_pretrained: str = "openai",
    encoder_device: Optional[str] = None,
    batch_size: int = 64,
) -> int:
    """Precompute and cache text embeddings for all unique questions.

    Writes ``<cache_dir>/query_embeddings__<encoder>__<pretrained>.npz``
    -- the per-encoder file ``run()`` looks up -- so subsequent ``run()``
    calls skip the text encoder forward on every question. A
    ``query_embeddings.npz`` in the directory does not count: it records
    no encoder.

    Parameters
    ----------
    encoder_device : str, optional
        Device for text encoding. Defaults to ``"cuda"`` if available,
        else ``"cpu"``. Heavy encoders (SigLIP2-SO400M) are ~50x faster
        on GPU; light ones (CLIP-B/32) are fine on CPU.
    batch_size : int
        Batch size for text encoding. Default 64.

    Returns the number of query embeddings written (0 if already cached).
    """
    cache_dir = Path(cache_dir)
    qe_path = cache_dir / _query_embeddings_filename(encoder_name, encoder_pretrained)
    if qe_path.exists():
        return 0

    seen = {}
    for ex in examples:
        if ex.question_id and ex.question_id not in seen:
            seen[ex.question_id] = ex.question

    if not seen:
        return 0

    # Auto-detect device
    if encoder_device is None:
        try:
            import torch
            encoder_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            encoder_device = "cpu"

    print(f"  Precomputing {len(seen)} query embeddings "
          f"({encoder_name} on {encoder_device})...")
    from ..encoders import CLIPEncoder

    encoder = CLIPEncoder(encoder_name, encoder_pretrained, encoder_device)
    qids = sorted(seen.keys())
    texts = [seen[q] for q in qids]

    # Batched encoding — huge speedup vs one-at-a-time
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Query embeddings"):
        batch = texts[i:i + batch_size]
        if hasattr(encoder, "encode_texts"):
            emb = encoder.encode_texts(batch)
        else:
            emb = np.stack([encoder.encode_text(t) for t in batch])
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        embeddings.append(emb)
    embeddings = np.concatenate(embeddings, axis=0)

    np.savez_compressed(
        str(qe_path),
        question_ids=np.array(qids),
        embeddings=embeddings,
    )
    print(f"  Saved {len(qids)} query embeddings → {qe_path.name}")

    # Release GPU memory — the encoder is no longer needed after
    # embeddings are written to disk.
    del encoder
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return len(qids)


# ---------------------------------------------------------------------------
# Base evaluator
# ---------------------------------------------------------------------------

class BaseEvaluator(ABC):
    """Shared benchmark orchestration.

    Simplest usage (subclass must implement the download hooks)::

        ev = ActivityNetEvaluator(cache_dir="./cache")
        r = ev.run(fs.create("efs", n_frames=16))

    Full control — all selection-axis configuration lives on the
    selector; the evaluator only carries dataset + answering-VLM state::

        ev = ActivityNetEvaluator(
            vlm="openai:Qwen/Qwen2.5-VL-7B-Instruct",
            video_dir="/path/to/activitynet/videos",
            cache_dir="./cache",
        )
        sel = fs.create(
            "efs",
            n_frames=16,
            candidate_frames=128,
            encoder="ViT-SO400M-16-SigLIP2-384",
            encoder_pretrained="webli",
        )
        r = ev.run(sel)

    See :meth:`__init__` for the full parameter reference.
    """

    # Override in subclass for per-subgroup accuracy reporting.
    subgroup_field: str = "duration"
    subgroup_values: tuple = ()

    # Override in subclass — default data_dir name under ~/.cache/frameselect/
    _default_data_dirname: str = "generic"

    # Sentinel to distinguish "cache_dir not passed" from "cache_dir=None".
    # We use a private string literal (not ``object()``) so the identity
    # check survives re-imports under tooling that instruments modules.
    # Any code that accidentally passes
    # the same literal will be misread as "not passed" — a practical
    # non-issue since nothing in the library ever constructs that
    # string on its own.
    _CACHE_UNSET = "__FRAMESELECT_CACHE_UNSET__"

    def __init__(
        self,
        video_dir: Optional["str | Path"] = None,
        annotations: Optional["str | Path | Iterable[Example]"] = None,
        cache_dir: "str | Path | None" = _CACHE_UNSET,
        *,
        vlm: Optional[str] = None,
        data_dir: Optional["str | Path"] = None,
        model_id: Optional[str] = None,
        image_size: Optional[int] = 512,
    ):
        """Construct an evaluator.

        Parameters
        ----------
        video_dir : str or Path, optional
            Directory containing video files. If omitted, the subclass'
            :meth:`_download_videos` hook fetches them.
        annotations : str, Path, or list[Example], optional
            Annotation file or pre-loaded examples. If omitted, the
            subclass' :meth:`_load_default_annotations` hook fetches
            them.
        cache_dir : str or Path, optional
            Embedding cache directory, one ``<video>.npz`` (+ JPEG
            sidecar) per video, so running ten selectors that share an
            encoder decodes + encodes each video once. The file name
            carries no encoder or pool size -- the *directory* does: use
            one ``cache_dir`` per encoder x candidate-pool size. A
            selector whose encoder, ``candidate_frames`` or
            ``candidate_fps`` does not match the cached files raises
            :class:`CacheMismatchError`. Pass ``None`` to disable.
        vlm : str, optional
            **Answering** VLM spec (e.g. ``"openai:Qwen/Qwen2.5-VL-7B-Instruct"``)
            — the VLM called by the evaluator's Step 3 to answer the
            benchmark's MCQ prompts. This has nothing to do with
            selection-time VLM calls; those live on VLM-based selectors
            and are configured via ``fs.create("<name>", vlm="bedrock:...")``.
        data_dir : str or Path, optional
            Root directory for auto-downloaded data (defaults to
            ``~/.cache/frameselect/<benchmark>``).
        model_id : str, optional
            Legacy override for the answering VLM's model ID. Prefer
            ``vlm=`` instead.
        image_size : int, optional
            Longest-edge pre-resize applied to every frame before
            base64 / PIL handoff to the VLM. Default 512 px, which keeps
            VLM prompts under
            ~12 k tokens on 16-frame runs, which is the difference
            between 4 req/s and 0.3 req/s on Qwen2.5-VL. Pass
            ``image_size=None`` to feed native-resolution frames to
            the VLM's own processor — only needed for OCR-heavy
            benchmarks on large-context models.

        Notes
        -----
        The evaluator no longer owns selection-time parameters
        (``encoder``, ``candidate_frames``, ``n_frames``). Those are
        **selector attributes** — set them via
        ``fs.create("efs", n_frames=16, encoder="...", ...)``.
        This keeps the two knobs from colliding and makes
        provenance clean: every :class:`SelectionResult` carries the
        exact selector configuration it was produced with.
        """
        # --- Resolve data directory ---
        if data_dir is None:
            data_dir = Path(f"~/.cache/frameselect/{self._default_data_dirname}").expanduser()
        else:
            data_dir = Path(data_dir).expanduser()
        data_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir = data_dir

        # --- Auto-download annotations if not provided ---
        if annotations is None:
            annotations = self._load_default_annotations(data_dir)

        # --- Auto-download videos if not provided ---
        if video_dir is None:
            video_dir = self._download_videos(data_dir)

        # --- Cache directory ---
        # Use a sentinel to distinguish "not provided" from "explicitly None".
        # If the user passes cache_dir=None explicitly, respect it (no cache).
        # If they don't pass it at all, auto-create under data_dir.
        if cache_dir == BaseEvaluator._CACHE_UNSET:
            cache_dir = data_dir / "cache" if data_dir is not None else None
        if cache_dir is not None:
            cache_dir = Path(cache_dir).expanduser()
            cache_dir.mkdir(parents=True, exist_ok=True)

        # --- Core state ---
        self.video_dir = Path(video_dir).expanduser()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        # Answering VLM: ``vlm=`` is the new canonical path. For now
        # we still accept ``model_id=`` for back-compat; it maps onto
        # the legacy ``_normalize_model_id`` behaviour (Bedrock prefix
        # handling). The ``vlm=`` spec takes priority.
        self._answering_vlm_spec = vlm
        # Resolve the backend and per-backend callables for this
        # evaluator instance. When ``vlm=`` is provided, it overrides
        # whatever the module-level auto-detection picked. We store
        # the resolved backend + callables on ``self`` so multiple
        # evaluator instances with different ``vlm=`` specs don't
        # clobber each other's module state.
        self._vlm_backend, self._build_mcq_message, self._call_fn = \
            self._resolve_backend_callables(vlm)
        self.model_id = _normalize_model_id_for_backend(
            _parse_answering_vlm_model(vlm) if vlm is not None else model_id,
            self._vlm_backend,
        )
        if self._vlm_backend == "bedrock" and self.model_id:
            # The Bedrock answering path only knows its catalog's models, so
            # reject an unknown id up front.
            from ..vlm.bedrock import MODEL_CATALOG, _canonical_model_name
            if _canonical_model_name(self.model_id) is None:
                raise ValueError(
                    f"Unknown Bedrock model {self.model_id!r}; expected one of "
                    f"{sorted(MODEL_CATALOG)} or their inference-profile IDs."
                )
        # Pre-resize longest edge before base64 encoding / HF PIL
        # handoff. Default 512 px. It is the single biggest knob on VLM prompt
        # length: a 1080p frame is ~2000 vision tokens on Qwen2.5-VL,
        # while 512 px is ~600 — 3× faster per prompt, and the
        # difference in MCQ accuracy is below noise. Pass
        # ``image_size=None`` to let the VLM's own processor see
        # native-resolution frames (required for OCR-heavy tasks on
        # large-context models).
        self._image_size = image_size

        # Accept: pre-parsed list[Example], or a path to an annotation
        # file which the subclass will load.
        if isinstance(annotations, (str, Path)):
            self.examples: list[Example] = list(
                self.load_annotations(str(annotations))
            )
        else:
            self.examples = list(annotations)

        # Optional cached query embeddings for query-aware methods.
        # Keyed under ``<cache_dir>/query_embeddings__<encoder>__<pretrained>.npz``
        # so different selectors' encoders don't clobber each other.
        # Populated lazily on ``run()``.
        self._query_embeddings_by_encoder: dict[tuple[str, str], dict] = {}

        # Dense per-candidate score curves captured during selection, as
        # ``{key: (all_scores, candidate_indices)}``. Repopulated per ``run()``.
        self._score_sink: dict = {}

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def precompute_cache(
        self,
        *,
        encoder: str = "ViT-B-32",
        encoder_pretrained: Optional[str] = None,
        candidate_fps: float = 1.0,
        candidate_frames: int = 128,
        n_gpus: Optional[int] = None,
        gpu_ids: Optional[list[int]] = None,
    ) -> int:
        """Precompute embeddings for every video in this benchmark.

        Call this once, before running any selectors, to amortise the
        video-decode + encoder-forward cost across all selectors that
        share the same encoder. Subsequent ``run()`` calls hit the
        cache.

        The encoder is chosen here — not on the evaluator — because
        different selectors may legitimately want different encoders
        (``ViT-B-32``, ``ViT-SO400M-16-SigLIP2-384``, etc.). Cache files
        are named by video only, so give each encoder x pool size its own
        evaluator ``cache_dir``: on a directory that already holds a
        video's file, this skips that video whatever encoder made it.

        Parameters
        ----------
        encoder : str
            OpenCLIP / SigLIP model name. Default ``"ViT-B-32"``.
        encoder_pretrained : str, optional
            Pretrained-weights tag. Defaults to ``"webli"`` for SigLIP
            encoders, ``"openai"`` otherwise.
        candidate_fps : float
            Sampling rate. Default 1.0 (1 FPS).
        candidate_frames : int
            Maximum candidate frames per video. Default 128.
        n_gpus : int, optional
            Number of GPUs to shard across. Defaults to every visible GPU
            (one CPU worker when there is none).
        gpu_ids : list[int], optional
            GPU indices as this process numbers them (positions in
            ``CUDA_VISIBLE_DEVICES`` when it is set). Defaults to
            ``list(range(n_gpus))``.

        Returns
        -------
        int
            Number of videos encoded.
        """
        if self.cache_dir is None:
            raise RuntimeError(
                "cache_dir must be set to use precompute_cache(). "
                "Pass cache_dir= to the evaluator constructor."
            )
        if encoder_pretrained is None:
            encoder_pretrained = "webli" if "SigLIP" in encoder else "openai"
        # Encode exactly this benchmark's videos, whatever their container:
        # globbing the directory for *.mp4 skipped ActivityNet's .mkv and
        # .webm files.
        videos = sorted({self.video_dir / ex.video_path for ex in self.examples})
        return precompute_embeddings_parallel(
            video_dir=[v for v in videos if v.exists()],
            cache_dir=self.cache_dir,
            encoder_name=encoder,
            encoder_pretrained=encoder_pretrained,
            candidate_fps=candidate_fps,
            max_candidates=candidate_frames,
            n_gpus=n_gpus,
            gpu_ids=gpu_ids,
        )

    @abstractmethod
    def load_annotations(self, path: str) -> Iterable[Example]:
        """Return an iterable of :class:`Example` from a dataset-specific file."""
        raise NotImplementedError

    def _download_videos(self, data_dir: Path) -> Path:
        """Download videos for this dataset. Override in subclass.

        Returns the directory containing the video files.
        Raises ``NotImplementedError`` with a helpful message if the
        subclass does not implement auto-download.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support auto-download.\n"
            f"Please provide video_dir= explicitly:\n"
            f"  ev = {type(self).__name__}(video_dir='/path/to/videos', ...)"
        )

    def _load_default_annotations(self, data_dir: Path) -> list[Example]:
        """Load annotations for this dataset. Override in subclass.

        Returns a list of :class:`Example`. Called when ``annotations``
        is not provided to the constructor.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support auto-download.\n"
            f"Please provide annotations= explicitly:\n"
            f"  ev = {type(self).__name__}(annotations='path/to/ann.json', ...)"
        )

    @classmethod
    def from_hub(cls, **kwargs) -> "BaseEvaluator":
        """Deprecated — use the constructor directly instead.

        The constructor now handles auto-download when ``video_dir``
        and ``annotations`` are omitted.
        """
        return cls(**kwargs)

    def _resolved_image_size(self) -> Optional[int]:
        """Return the effective image size for VLM calls, or ``None``
        for passthrough (let the VLM's processor handle resolution).

        Set via the ``image_size=`` constructor param (default 512 px).
        Returning ``None`` lets the VLM's own processor handle native
        resolution — useful for OCR-heavy workloads on large-context
        models, but ~3× slower on high-res video benchmarks where
        frames come from 1080p/4K source.
        """
        return self._image_size

    @staticmethod
    def _resolve_backend_callables(vlm_spec: Optional[str]):
        """Return ``(backend, build_mcq_fn, call_fn)`` for this evaluator.

        If ``vlm_spec`` carries a backend prefix (e.g. ``"openai:Qwen/..."``),
        the backend is derived from it; an unknown prefix raises
        ``ValueError`` instead of silently routing to another provider.
        Without a prefix (or without a spec) it falls back to the
        module-level auto-detected ``_VLM_BACKEND``.

        Returns instance-local references so multiple evaluators with
        different ``vlm=`` specs don't clobber each other.
        """
        backend = _split_vlm_spec(vlm_spec)[0] if vlm_spec else None
        if backend is None:
            backend = _get_vlm_backend()

        _, _build, _call = _get_backend_symbols(backend)
        return backend, _build, _call

    def _maybe_start_answering_vlm(self, verbose: bool = True) -> None:
        """If the answering VLM is an ``openai:<model>`` spec and no
        server is reachable at the resolved URL, start a managed vLLM
        process so ``ev.run(sel)`` works end-to-end without a second
        terminal.

        Mirrors :meth:`BaseSelector.warmup` (which does the same thing
        for selection-time VLMs). The managed server is stopped when
        this evaluator is garbage-collected.

        No-op when:
        * the evaluator didn't get ``vlm="openai:..."`` at construction
        * a server is already reachable (env var, running vLLM, etc.)
        * the spec targets a non-openai backend (bedrock / anthropic /
          hf paths don't need a local server)
        """
        spec = getattr(self, "_answering_vlm_spec", None)
        if spec is None or not spec.lower().startswith("openai:"):
            return
        if getattr(self, "_managed_vlm_server", None) is not None:
            return  # already started by a previous run() on this instance

        from ..vlm.openai import _default_base_url, _probe_local_vllm
        base_url = _default_base_url() or "http://localhost:8000/v1"
        from urllib.parse import urlparse
        p = urlparse(base_url)
        host, port = p.hostname or "localhost", p.port or 8000

        if _probe_local_vllm(host=host, port=port) is not None:
            return  # user or a previous step already started a server

        model = spec.split(":", 1)[1]
        if verbose:
            print(
                f"  No vLLM server reachable at {host}:{port}; "
                f"auto-starting one for {model}..."
            )
        from ..vlm.server import VLMServer
        self._managed_vlm_server = VLMServer.start(
            model=model,
            port=port,
            verbose=verbose,
        )
        # Backend callables (``self._build_mcq_message``,
        # ``self._call_fn``) were resolved in ``__init__`` from the
        # ``vlm=`` spec. ``self.model_id`` was also set there. The
        # only job of this method is to bring up the server process.

    def close(self, verbose: bool = False) -> None:
        """Stop the managed vLLM server (if any) started by this evaluator.

        Explicit, idempotent teardown for the vLLM process that
        :meth:`_maybe_start_answering_vlm` may have spawned. ``run()``
        calls this after answering completes; :meth:`__del__` also calls
        it as a backstop for callers that never invoke ``close()``.
        Safe to call more than once.
        """
        srv = getattr(self, "_managed_vlm_server", None)
        if srv is not None:
            try:
                srv.stop(verbose=verbose)
            except Exception as exc:
                logger.debug(
                    "managed vLLM server stop failed: %s: %s",
                    type(exc).__name__, exc,
                )
            finally:
                self._managed_vlm_server = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __del__(self):
        """Stop the managed vLLM server (if any) on evaluator teardown."""
        try:
            self.close(verbose=False)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        selector,
        *,
        n_workers: int = 8,
        verbose: bool = True,
        selections_from: Optional[str] = None,
        selection_query_map: Optional[dict] = None,
    ) -> EvaluationResult:
        """Run the full benchmark (select + score) for ``selector``.

        Parameters
        ----------
        selector : BaseSelector
            The frame-selector to evaluate. All selection-time
            configuration — ``n_frames``, ``candidate_frames``,
            ``candidate_fps``, ``encoder``, ``encoder_pretrained`` —
            lives on the selector. Configure it at construction:
            ``fs.create("efs", n_frames=16, encoder="ViT-SO400M-...")``.
        n_workers : int
            Parallelism for VLM calls. Default 8.
        verbose : bool
            Print progress. Default True.
        selections_from : str, optional
            Path to a saved selections JSON for replay mode.
        selection_query_map : dict, optional
            Optional ``{question_id: query_str}`` override consulted ONLY
            at frame-selection time. When given, the selector sees the
            mapped query (e.g. an English translation) while the answering
            VLM, options and ground truth continue to use the native
            ``ex.question``, e.g. to select with a translated query while
            answering in the original language. When
            an entry is missing for a ``question_id``, the native question
            is used. Setting this also disables the query-embedding cache
            (the cached embeddings are for the native question).
        """
        try:
            return self._run_mcq(selector, n_workers=n_workers, verbose=verbose,
                                 selections_from=selections_from,
                                 selection_query_map=selection_query_map)
        finally:
            # Free the shared encoder and any model the selector loaded onto
            # the GPU, and tear down any vLLM server the run auto-started --
            # on failure too, so the next selector starts with free GPUs.
            self._release_gpu_state(selector)
            self.close()

    def _run_mcq(self, selector, *, n_workers, verbose, selections_from,
                 selection_query_map) -> EvaluationResult:
        self._selection_query_map = selection_query_map
        # Selection-axis controls all come from the selector.
        n_frames = getattr(selector, "n_frames", 16)
        candidate_fps = getattr(selector, "candidate_fps", 1.0)
        t_start = time.perf_counter()
        provenance = self._run_provenance(
            selector, selections_from=selections_from,
            selection_query_override=bool(selection_query_map),
        )

        if verbose:
            print(f"VLM backend: {self._vlm_backend}")

        # Bedrock answering calls are network-bound, so they run on a
        # thread pool (no forking) -- see :meth:`_run_vlm`.

        # --- Step 1: selection ---
        sel_frames, sel_indices, sel_lat_ms, n_errors = self._run_selection(
            selector, n_frames, candidate_fps, n_workers, verbose, selections_from
        )

        # --- Build rows from per-key selections ---
        rows = self._build_rows(selector, sel_frames, sel_indices, sel_lat_ms)
        df = pd.DataFrame(rows)

        # --- Step 2: base64 messages ---
        n_threads = max(4, n_workers * 4)
        if verbose:
            print(f"Step 2/3: Encoding the frames of {len(rows)} questions as base64 ({n_threads} threads)...")
        rows_list = [row for _, row in df.iterrows()]
        msgs = [None] * len(rows_list)

        img_sz = self._resolved_image_size()

        def _build_one(row):
            return self._build_mcq_message(row["frames"], row["question"], row["options"],
                                     image_size=img_sz)

        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            futures = {ex.submit(_build_one, r): i for i, r in enumerate(rows_list)}
            for f in tqdm(
                as_completed(futures), total=len(futures),
                desc="Build messages", disable=not verbose,
            ):
                msgs[futures[f]] = f.result()
        df["messages"] = msgs
        df = _drop_frames_column(df, context="MCQ evaluation")

        # --- Step 3: VLM ---
        # Start the answering VLM here rather than at the top of
        # run(): selection can itself need GPUs (CLIP-based encoders, or a
        # custom selector that calls a VLM). Booting the answering server
        # upfront would (a) fight selection for GPU memory, (b)
        # deadlock when a selection-time VLM tries to grab GPUs vLLM
        # already holds, and (c) waste ~2 min of startup when the
        # user is replaying from a selections cache. Starting it
        # here, after selection is complete, means the answering
        # VLM only comes up when it's actually needed.
        self._maybe_start_answering_vlm(verbose=verbose)

        df["pred"] = self._run_vlm(df, n_workers, verbose)
        if len(df) and bool((df["pred"] == FAILED_ANSWER).all()):
            # Not a 0% result: the answering VLM never replied (dead server,
            # wrong model name, missing credentials). Fail so nothing is saved.
            raise RuntimeError(
                f"Every answering-VLM call failed ({len(df)} of {len(df)}; "
                f"backend={self._vlm_backend}, model={self.model_id}). Check that "
                "the server is reachable and the model name is right; the warnings "
                "above give the error type."
            )

        # --- Score + report ---
        # A prediction is correct only when it matches the ground truth
        # AND is not the VLM failure sentinel (a failed VLM call returns
        # ``FAILED_ANSWER`` rather than a spurious "A" — it must count as
        # incorrect, never as a lucky match).
        df["correct"] = (df["pred"] == df["gt"]) & (df["pred"] != FAILED_ANSWER)
        correct = int(df["correct"].sum())
        # Denominator is EVERY example, not just the ones that produced a
        # scored row: a missing / failed selection (dropped from ``df``)
        # counts as incorrect, so failures shrink accuracy — not the
        # denominator. This is the correct behaviour for a benchmark lib.
        total = len(self.examples)
        accuracy = round(correct / total * 100, 2) if total > 0 else 0.0

        latency = self._latency_stats(sel_lat_ms, selector.query_aware)
        subgroup_acc = self._subgroup_accuracy(df)

        if verbose:
            self._print_report(accuracy, correct, total, subgroup_acc, latency)
            n_unscored = total - len(df)
            if n_unscored > 0:
                print(f"  ({n_unscored} example(s) had no valid selection and "
                      f"count as incorrect; {n_errors} selection error(s))")

        elapsed = time.perf_counter() - t_start

        return EvaluationResult(
            accuracy=accuracy,
            correct=correct,
            total=total,
            n_frames=n_frames,
            method=getattr(selector, 'method_name', selector.__class__.__name__),
            model_id=self.model_id,
            vlm_backend=self._vlm_backend,
            latency=latency,
            subgroup_accuracy=subgroup_acc,
            n_errors=n_errors,
            df=df,
            elapsed_sec=round(elapsed, 2),
            provenance=provenance,
        )

    def _run_provenance(self, selector, *, answering: bool = True,
                        selections_from: Optional[str] = None,
                        **extra) -> dict:
        """The configuration behind a run, saved as ``"provenance"`` by
        :meth:`EvaluationResult.save` so a result JSON documents itself.

        The selector half mirrors the provenance of a :class:`SelectionResult`
        (``selector``, ``seed``, ``hyperparams`` = ``metadata["hyperparams"]``,
        ...) but records the *configured* ``encoder`` / ``encoder_pretrained``
        attributes -- the ones this evaluator's cache lookup and precompute use
        -- so encoder-free selectors (uniform) carry a value they
        ignore. The evaluator half is the answering VLM and ``image_size``;
        with ``answering=False`` (no VLM answers anything) those keys are
        ``None``. ``extra`` keys are merged in.
        """
        from .. import __version__

        hyper = getattr(selector, "_hyperparams", None)
        try:
            hyperparams = dict(hyper()) if callable(hyper) else {}
        except Exception as exc:
            logger.debug("selector._hyperparams() failed: %s: %s",
                         type(exc).__name__, exc)
            hyperparams = {}
        prov = {
            "frameselect_version": __version__,
            "evaluator": type(self).__name__,
            "selector": getattr(selector, "method_name", type(selector).__name__),
            "seed": getattr(selector, "seed", None),
            "n_frames": getattr(selector, "n_frames", 16),
            "candidate_frames": getattr(selector, "candidate_frames", None),
            "candidate_fps": getattr(selector, "candidate_fps", 1.0),
            "encoder": getattr(selector, "encoder", None),
            "encoder_pretrained": getattr(selector, "encoder_pretrained", None),
            "text_encoder": getattr(selector, "text_encoder", None),
            "selection_vlm": getattr(selector, "_vlm_spec", None),
            "hyperparams": hyperparams,
            "answering_vlm": self._answering_vlm_spec if answering else None,
            "answering_model_id": self.model_id if answering else None,
            "vlm_backend": self._vlm_backend if answering else None,
            "image_size": self._image_size if answering else None,
            # Replay runs re-use saved picks: name the file (not its path).
            "selections_from": Path(selections_from).name if selections_from else None,
        }
        prov.update(extra)
        return prov

    @staticmethod
    def _release_gpu_state(selector) -> None:
        """Clear GPU state held by the selector after a run.

        Steps:

        1. Stop any vLLM subprocess the selector spawned.
        2. Call ``.release()`` on encoder/model handles that know how
           to drop their GPU allocation (``CLIPEncoder`` has explicit
           cleanup; other handles just get unreferenced).
        3. Null the selector's attribute references so the next
           ``warmup()`` starts from scratch.
        4. Run ``gc.collect()`` + ``torch.cuda.empty_cache()``.

        This ensures the next method's ``warmup()`` — which may start a
        vLLM server expecting ~20GB of free memory — finds a clean GPU.
        """
        # (1) Stop managed vLLM subprocess if present.
        stopper = getattr(selector, "_stop_managed_server", None)
        if callable(stopper):
            try:
                stopper()
            except Exception as exc:
                logger.debug("selector._stop_managed_server() failed: %s: %s",
                             type(exc).__name__, exc)

        # (2) Explicit release for encoder/model handles that know how.
        # (3) Null every attribute reference in one sweep.
        for attr in ("_encoder", "_clip", "_dino", "_model", "_processor",
                     "_client", "_backend"):
            if hasattr(selector, attr):
                obj = getattr(selector, attr, None)
                release = getattr(obj, "release", None)
                if callable(release):
                    try:
                        release()
                    except Exception as exc:
                        logger.debug("release() on %s.%s failed: %s: %s",
                                     type(selector).__name__, attr,
                                     type(exc).__name__, exc)
                try:
                    setattr(selector, attr, None)
                except Exception as exc:
                    logger.debug("nulling %s.%s failed: %s: %s",
                                 type(selector).__name__, attr,
                                 type(exc).__name__, exc)

        # (4) Final GC + CUDA cache empty.
        try:
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as exc:
            logger.debug("GPU cache cleanup failed: %s: %s",
                         type(exc).__name__, exc)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _unique_videos(self) -> dict:
        out = {}
        for ex in self.examples:
            if ex.video_path not in out:
                out[ex.video_path] = ex
        return out

    def _run_selection(self, selector, n_frames, candidate_fps,
                       n_workers, verbose, selections_from):
        use_cache = self.cache_dir is not None
        replay_selections = None
        replay_latencies = {}

        if selections_from is not None:
            if verbose:
                print(f"Step 1/3: REPLAY — loading selections from {selections_from}")
            replay_selections = _load_saved_selections(selections_from)
            try:
                with open(selections_from, "r") as _f:
                    _src = json.load(_f)
                for _rec in _src.get("per_video", []):
                    _key = (_rec["video"], str(_rec.get("question_id", "")))
                    if _rec.get("selection_latency_ms") is not None:
                        replay_latencies[_key] = float(_rec["selection_latency_ms"])
            except Exception as exc:
                logger.debug("could not parse replay latencies from %s: %s: %s",
                             selections_from, type(exc).__name__, exc)
        elif verbose:
            unique_count = len(self._unique_videos())
            if use_cache:
                cached = sum(
                    1 for v in self._unique_videos()
                    if _cache_npz_path(self.cache_dir, v).exists()
                )
                print(f"Step 1/3: Selecting {n_frames} frames "
                      f"({'per-question' if selector.query_aware else 'per-video'}, "
                      f"cache: {cached}/{unique_count})...")
            else:
                print(f"Step 1/3: Selecting {n_frames} frames "
                      f"({'per-question' if selector.query_aware else 'per-video'}, no cache)...")

        sel_frames, sel_indices, sel_lat = {}, {}, {}
        errors = 0
        _video_cache = {}
        # Fresh per run(), so no curve from a previous selector carries over.
        self._score_sink = {}

        if replay_selections is not None:
            errors += self._run_replay(
                selector, replay_selections, replay_latencies,
                sel_frames, sel_indices, sel_lat,
                use_cache, n_workers, verbose,
            )
        else:
            errors += self._run_fresh(
                selector, n_frames, candidate_fps,
                sel_frames, sel_indices, sel_lat,
                use_cache, n_workers, verbose, _video_cache,
            )

        if verbose:
            print(f"  Done: {len(sel_frames)} selections, {errors} errors")
            if sel_lat:
                lats = np.array(list(sel_lat.values()))
                print(f"  Latency per selection: mean={lats.mean():.1f} ms, "
                      f"p50={np.percentile(lats, 50):.1f} ms, "
                      f"p95={np.percentile(lats, 95):.1f} ms")

        return sel_frames, sel_indices, sel_lat, errors

    def _iter_selection_units(self, selector) -> list[Example]:
        """Yield one unit of selection work per selector-semantics:
        per-question for query-aware, per-video for query-agnostic.
        """
        if selector.query_aware:
            return list(self.examples)
        return list(self._unique_videos().values())

    def _run_replay(self, selector, replay_selections, replay_latencies,
                    sel_frames, sel_indices, sel_lat, use_cache,
                    n_workers, verbose):
        if not use_cache:
            # Replay requires the precomputed per-video frame cache so we
            # can load JPEGs for the saved abs_indices.
            raise RuntimeError(
                "Replay mode requires cache_dir= (to load JPEGs for saved indices)."
            )

        iterable = self._iter_selection_units(selector)

        # Query-agnostic replay may key saved selections by
        # ``(video, question_id)`` even though selection is per-video.
        # Build a ``video → indices`` fallback map ONCE (was an O(N²)
        # linear scan of ``replay_selections`` per example).
        video_to_saved: dict = {}
        if not selector.query_aware:
            for (rv, _qid), idxs in replay_selections.items():
                video_to_saved.setdefault(rv, idxs)

        def _replay_one(ex):
            qid = ex.question_id
            key = qid if selector.query_aware else ex.video_path
            cache_path = str(_cache_npz_path(self.cache_dir, ex.video_path))
            replay_key = (ex.video_path, str(qid))
            saved = replay_selections.get(replay_key)
            if saved is None and not selector.query_aware:
                saved = video_to_saved.get(ex.video_path)
            if saved is None or not os.path.exists(cache_path):
                return key, None, None, replay_key
            cache = _load_cache(cache_path)
            abs_idx, frames = _load_frames_by_abs_indices(cache, saved)
            return key, abs_idx, frames, replay_key

        n_io = max(n_workers * 2, 16)
        errors = 0
        with ThreadPoolExecutor(max_workers=n_io) as ex:
            futs = [ex.submit(_replay_one, e) for e in iterable]
            for fut in tqdm(
                as_completed(futs), total=len(futs),
                desc="Loading saved frames (replay)", disable=not verbose,
            ):
                key, abs_idx, frames, replay_key = fut.result()
                if frames is None:
                    errors += 1
                    continue
                sel_frames[key] = frames
                sel_indices[key] = abs_idx.tolist()
                if replay_key in replay_latencies:
                    sel_lat[key] = replay_latencies[replay_key]
        return errors

    def _check_first_cached_video(self, selector) -> None:
        """Check the first cached video against ``selector`` (fail fast).

        Raises :class:`CacheMismatchError` when ``cache_dir`` was built for
        another encoder or candidate pool, before any precompute or
        selection runs. Every file is checked again when it is loaded. An
        unreadable file is skipped here; selection reports it for its video.
        """
        for v in self._unique_videos():
            p = _cache_npz_path(self.cache_dir, v)
            if not p.exists():
                continue
            try:
                cache = _load_cache(str(p))
            except Exception as exc:
                logger.debug("skipping unreadable cache file %s: %s: %s",
                             p, type(exc).__name__, exc)
                continue
            check_cache_matches_selector(cache, selector, str(p))
            return

    def _load_query_embeddings(self, selector, encoder: str,
                               pretrained) -> Optional[dict]:
        """``{question_id: embedding}`` for ``encoder`` in this cache_dir.

        Reads ``query_embeddings__<encoder>__<pretrained>.npz``, computing it
        with :func:`precompute_query_embeddings` when absent. The older
        single-file ``query_embeddings.npz`` records no encoder, so it is
        only used for the default ViT-B-32 / openai encoder and only when
        its width matches.
        """
        qe_path = self.cache_dir / _query_embeddings_filename(encoder, pretrained)
        legacy = self.cache_dir / _LEGACY_QUERY_EMBEDDINGS
        if (not qe_path.exists() and legacy.exists()
                and encoder == "ViT-B-32" and pretrained in (None, "openai")):
            data = _read_query_embeddings(legacy)
            dim = len(next(iter(data.values()))) if data else None
            want = encoder_output_dim(encoder)
            if want is None or dim == want:
                return data
            logger.warning(
                "Ignoring %s: its %s-d query embeddings do not match %s (%s-d); "
                "computing %s instead.", legacy, dim, encoder, want, qe_path.name,
            )
        if not qe_path.exists():
            # Encode queries on the SELECTOR's device, not the global
            # ``"cuda"`` default (== cuda:0), which may be busy (for example
            # with a local vLLM server).
            precompute_query_embeddings(
                examples=self.examples,
                cache_dir=self.cache_dir,
                encoder_name=encoder,
                encoder_pretrained=pretrained,
                encoder_device=getattr(selector, "device", None),
            )
        return _read_query_embeddings(qe_path) if qe_path.exists() else None

    def _run_fresh(self, selector, n_frames, candidate_fps,
                   sel_frames, sel_indices, sel_lat, use_cache,
                   n_workers, verbose, _video_cache):
        iterable = self._iter_selection_units(selector)
        import threading
        _video_cache_lock = threading.Lock()

        # Prevent nested parallelism: sklearn/numpy/BLAS spawn their own
        # threads via OpenMP. Inside a ThreadPoolExecutor this causes a
        # futex deadlock. Force single-threaded BLAS/OMP.
        import os as _os
        _os.environ["OMP_NUM_THREADS"] = "1"
        _os.environ["OPENBLAS_NUM_THREADS"] = "1"
        _os.environ["MKL_NUM_THREADS"] = "1"
        _os.environ["NUMEXPR_NUM_THREADS"] = "1"
        try:
            from threadpoolctl import threadpool_limits
            threadpool_limits(limits=1)
        except ImportError:
            pass

        # For CPU-bound cached-embedding selection, threads give zero
        # speedup (GIL + BLAS contention) and cause deadlocks. Only use
        # threads when the selector opts in (``parallel_selection``).
        _parallel = (
            n_workers > 1
            and getattr(selector, "parallel_selection", False)
        )

        # Check once whether this selector uses an encoder at all.
        _selector_uses_embeddings = (
            use_cache
            and selector._cache_config() is not None
        )

        # Phase A — Precompute embeddings FIRST (uses all GPUs, then frees them).
        # Must happen before warmup() so the vLLM server (if any) doesn't
        # occupy the GPUs that the encoder needs.
        #
        # The encoder is pulled off the selector — the selector owns its
        # encoding config (``self.encoder`` / ``self.encoder_pretrained``),
        # not the evaluator. The cache itself is one ``<video>.npz`` per
        # video, so the *directory* fixes the encoder and candidate pool:
        # each file is checked against the selector before use (see
        # ``check_cache_matches_selector``), and the first one is checked
        # here, before precompute could add this selector's embeddings to a
        # directory built for another encoder or pool size.
        _sel_encoder = getattr(selector, "encoder", "ViT-B-32")
        _sel_encoder_pretrained = getattr(selector, "encoder_pretrained", "openai")
        _sel_candidate_frames = getattr(selector, "candidate_frames", 128)
        if use_cache:
            self._check_first_cached_video(selector)
        if _selector_uses_embeddings and use_cache:
            unique_videos = list(self._unique_videos().keys())
            n_missing = sum(
                1 for v in unique_videos
                if not _cache_npz_path(self.cache_dir, v).exists()
            )
            if n_missing > 0:
                if verbose:
                    print(f"  Precomputing embeddings for {n_missing} videos "
                          f"(parallel across GPUs, encoder={_sel_encoder})...")
                # Pass only the specific video paths needed, not the whole dir
                video_paths_needed = [
                    str(self.video_dir / v)
                    for v in unique_videos
                    if not _cache_npz_path(self.cache_dir, v).exists()
                ]
                precompute_embeddings_parallel(
                    video_dir=video_paths_needed,
                    cache_dir=self.cache_dir,
                    encoder_name=_sel_encoder,
                    encoder_pretrained=_sel_encoder_pretrained,
                    candidate_fps=candidate_fps or 1.0,
                    max_candidates=_sel_candidate_frames,
                )
                # Spawned processes have exited — GPUs are now free.

        # Also precompute query embeddings (text-only, fast on CPU).
        # Only needed for query-aware selectors. Skipped for uniform/kmeans.
        # Keyed per-encoder so two selectors with different encoders
        # don't clobber each other's text embeddings.
        # A decoupled text encoder (``selector.text_encoder``, e.g. M-CLIP)
        # or a selection-time query override means the precomputed English
        # query embeddings (built from the IMAGE encoder's text tower on the
        # native question) are WRONG for selection — using them would
        # silently bypass the selector's own text encoder / the override. In
        # those cases skip the query-
        # embedding cache entirely so each selection encodes its query live
        # through the selector's own ``_get_encoder()`` (correct + cheap).
        # Selectors with no embedding fast path never read query
        # embeddings, so they don't trigger a text-encoder precompute either.
        _text_encoder_active = bool(getattr(selector, "text_encoder", None))
        _query_override_active = bool(getattr(self, "_selection_query_map", None))
        # A selector that reads the answer choices encodes question + choice
        # texts itself, so the question-only embeddings would go unused.
        _uses_choices = bool(getattr(selector, "uses_answer_choices", False))
        _query_embeddings_cached: dict = {}
        if (
            use_cache
            and getattr(selector, "query_aware", False)
            and _reads_embeddings(selector)
            and not _text_encoder_active
            and not _query_override_active
            and not _uses_choices
            and any(ex.question_id for ex in self.examples)
        ):
            _qe_key = (_sel_encoder, _sel_encoder_pretrained)
            cached = self._query_embeddings_by_encoder.get(_qe_key)
            if cached is None:
                cached = self._load_query_embeddings(
                    selector, _sel_encoder, _sel_encoder_pretrained
                )
                if cached is not None:
                    self._query_embeddings_by_encoder[_qe_key] = cached
            _query_embeddings_cached = cached or {}
        # Width of the query embeddings handed to the selector, so a cache
        # file of another width is refused instead of failing on a matmul.
        _query_dim = (
            int(np.asarray(next(iter(_query_embeddings_cached.values()))).shape[-1])
            if _query_embeddings_cached else None
        )

        # Phase B — warmup() AFTER precompute.
        # For a VLM-based selector with vlm="openai:...", this starts the
        # vLLM server on the now-free GPUs.
        if _parallel and hasattr(selector, "warmup"):
            try:
                selector.warmup()
            except Exception as exc:
                logger.warning("selector.warmup() failed: %s: %s",
                               type(exc).__name__, exc)

        # Performance: selectors that build a CLIPEncoder via
        # ``self._get_encoder`` lazy-init it thread-safely, so
        # ThreadPoolExecutor workers share one encoder instance
        # rather than racing on separate loads. No cross-selector
        # injection is needed anymore — the selector carries its
        # own encoder config, and each cache file is checked against it
        # (``check_cache_matches_selector``) when it is loaded.

        def _do_one(ex):
            key = ex.question_id if selector.query_aware else ex.video_path
            cache_path = (
                str(_cache_npz_path(self.cache_dir, ex.video_path))
                if use_cache else None
            )
            video_path = str(self.video_dir / ex.video_path)
            # Selection query: normally the native question, but a
            # selection_query_map overrides
            # it for SELECTION ONLY — the answering side (_build_rows) keeps
            # ex.question. Missing entries fall back to the native question.
            if selector.query_aware:
                _qmap = getattr(self, "_selection_query_map", None)
                query = (_qmap or {}).get(ex.question_id, ex.question) if _qmap else ex.question
                if _uses_choices and ex.options:
                    query = "\n".join([query, *ex.options])
            else:
                query = None
            # query_emb is the precomputed English embedding; it is only
            # populated when neither a decoupled text encoder nor a query
            # override is active (see the precompute guard above). Otherwise
            # it stays None so the selector encodes ``query`` live.
            query_emb = (
                _query_embeddings_cached.get(ex.question_id)
                if selector.query_aware else None
            )
            try:
                if cache_path and os.path.exists(cache_path):
                    with _video_cache_lock:
                        if ex.video_path not in _video_cache:
                            loaded = _load_cache(cache_path)
                            # Raises CacheMismatchError, which aborts the run
                            # (re-raised below) instead of scoring one error
                            # per item: the whole directory is the problem.
                            check_cache_matches_selector(
                                loaded, selector, cache_path, query_dim=_query_dim,
                            )
                            _video_cache[ex.video_path] = loaded
                        cache = _video_cache[ex.video_path]
                    t0 = time.perf_counter()
                    # ``strict=True``: if neither the JPEG sidecar nor the
                    # source video is available, raise instead of silently
                    # returning black frames — the example is then counted
                    # as a selection error (and scores incorrect).
                    indices, selected = _select_from_cache(
                        cache, selector, n_frames, query, query_emb,
                        video_path=video_path, strict=True,
                        score_sink=self._score_sink, sink_key=key,
                    )
                    dt = (time.perf_counter() - t0) * 1000.0
                    return key, indices.tolist(), selected, dt, None
                if os.path.exists(video_path):
                    t0 = time.perf_counter()
                    indices, frames_out = _select_from_video(
                        video_path, selector, n_frames, query, candidate_fps,
                        max_candidates=_sel_candidate_frames,
                        score_sink=self._score_sink, sink_key=key,
                    )
                    dt = (time.perf_counter() - t0) * 1000.0
                    return key, indices.tolist(), frames_out, dt, None
                return key, None, None, None, f"missing video: {video_path}"
            except CacheMismatchError:
                raise
            except Exception as exc:
                return key, None, None, None, f"{type(exc).__name__}: {exc}"

        errors = 0
        if _parallel:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futs = {ex.submit(_do_one, e): e for e in iterable}
                for fut in tqdm(
                    as_completed(futs), total=len(futs),
                    desc="Frame selection", disable=not verbose,
                ):
                    try:
                        key, idxs, fr, dt, err = fut.result()
                    except CacheMismatchError:
                        for f in futs:
                            f.cancel()   # don't start selections bound to fail
                        raise
                    if err:
                        errors += 1
                        if verbose:
                            tqdm.write(f"  ERR {key}: {err}")
                        continue
                    sel_frames[key] = fr
                    sel_indices[key] = idxs
                    if dt is not None:
                        sel_lat[key] = dt
        else:
            for ex_ in tqdm(iterable, desc="Frame selection", disable=not verbose):
                key, idxs, fr, dt, err = _do_one(ex_)
                if err:
                    errors += 1
                    if verbose:
                        tqdm.write(f"  ERR {key}: {err}")
                    continue
                sel_frames[key] = fr
                sel_indices[key] = idxs
                if dt is not None:
                    sel_lat[key] = dt

        # No evaluator-owned shared encoder exists anymore — the
        # selector owns its encoder and will be torn down (if
        # appropriate) in :meth:`_release_gpu_state` after the run.
        return errors

    def _build_rows(self, selector, sel_frames, sel_indices, sel_lat):
        rows = []
        for ex in self.examples:
            key = ex.question_id if selector.query_aware else ex.video_path
            frames = sel_frames.get(key)
            if frames is None:
                continue
            rows.append({
                "video": ex.video_path,
                "question_id": ex.question_id,
                "question": ex.question,
                "options": ex.options,
                "gt": ex.answer,
                self.subgroup_field: getattr(ex, self.subgroup_field, ""),
                "frames": frames,
                "selected_indices": sel_indices.get(key, []),
                "selection_latency_ms": sel_lat.get(key),
            })
        return rows

    def _run_vlm(self, df, n_workers, verbose) -> list[str]:
        backend = self._vlm_backend
        call_fn_base = self._call_fn

        if backend == "bedrock":
            from ..vlm.bedrock import call_haiku_parallel, _get_all_endpoints
            df["region_index"] = df.index % max(1, len(_get_all_endpoints()))
            n_http = max(n_workers * 4, 16)
            if verbose:
                print(f"Step 3/3: Calling Bedrock (model={self.model_id}) "
                      f"({len(df)} questions, {n_http} concurrent HTTP)...")
            rows_list = [row for _, row in df.iterrows()]
            preds = [FAILED_ANSWER] * len(rows_list)
            call_fn = partial(call_haiku_parallel, model_id=self.model_id)
            with ThreadPoolExecutor(max_workers=n_http) as ex:
                futures = {ex.submit(call_fn, r): i for i, r in enumerate(rows_list)}
                for f in tqdm(as_completed(futures), total=len(futures),
                              desc="Bedrock", disable=not verbose):
                    preds[futures[f]] = f.result()
            return preds

        if backend == "openai":
            # Client-side thread-pool width for HTTP fan-out. Should
            # not exceed the server's ``max_num_seqs`` (default 32 in
            # :meth:`VLMServer.start`) — submitting more just queues
            # them and drives up p95 latency without gaining
            # throughput. When the server is launched with a larger ``max_num_seqs``
            # (and lighter per-prompt token budget, e.g. ``image_size``
            # ≤ 336 + a ``max_pixels`` cap), raise this to match via
            # ``FRAMESELECT_VLM_HTTP_CONCURRENCY``. Default stays 32.
            _http_cap = int(os.environ.get("FRAMESELECT_VLM_HTTP_CONCURRENCY", "32"))
            n_http = min(max(n_workers * 4, 32), _http_cap)
            if verbose:
                print(f"Step 3/3: Calling OpenAI-compatible VLM ({len(df)} "
                      f"questions, {n_http} concurrent, model={self.model_id})...")
            rows_list = [row for _, row in df.iterrows()]
            preds = [FAILED_ANSWER] * len(rows_list)
            call_fn = partial(call_fn_base, model_id=self.model_id)
            with ThreadPoolExecutor(max_workers=n_http) as ex:
                futures = {ex.submit(call_fn, r): i for i, r in enumerate(rows_list)}
                for f in tqdm(as_completed(futures), total=len(futures),
                              desc="VLM", disable=not verbose):
                    preds[futures[f]] = f.result()
            return preds

        if backend == "hf":
            if verbose:
                print(f"Step 3/3: Calling HuggingFace VLM ({len(df)} questions, "
                      f"sequential, model={self.model_id})...")
            preds = []
            for _, row in tqdm(df.iterrows(), total=len(df),
                               desc="HF-VLM", disable=not verbose):
                preds.append(call_fn_base(row, model_id=self.model_id))
            return preds

        # Anthropic fallback
        if verbose:
            print(f"Step 3/3: Calling the Anthropic API (model={self.model_id}, {len(df)} questions, sequential)...")
        preds = []
        for _, row in tqdm(df.iterrows(), total=len(df),
                           desc="Anthropic", disable=not verbose):
            preds.append(call_fn_base(row, model_id=self.model_id))
        return preds

    def _latency_stats(self, sel_lat: dict, query_aware: bool) -> dict:
        if not sel_lat:
            return {}
        lats = np.array(list(sel_lat.values()))
        p50 = round(float(np.percentile(lats, 50)), 2)
        return {
            "per_selection_mean_ms": round(float(lats.mean()), 2),
            "per_selection_median_ms": p50,
            # Alias of the median above, kept in sync.
            "per_selection_p50_ms": p50,
            "per_selection_p95_ms": round(float(np.percentile(lats, 95)), 2),
            "per_selection_std_ms": round(float(lats.std()), 2),
            "total_selection_sec": round(float(lats.sum() / 1000.0), 2),
            "n_measured": int(len(lats)),
            "granularity": "per_question" if query_aware else "per_video",
        }

    def _subgroup_accuracy(self, df) -> dict:
        # Same denominator as the headline ``accuracy`` in ``run()``: every
        # example in the subgroup, so a failed selection (no row in ``df``)
        # counts as incorrect in its subgroup too, and the subgroup values
        # average back to the headline.
        totals: dict = {}
        for ex in self.examples:
            v = getattr(ex, self.subgroup_field, "")
            totals[v] = totals.get(v, 0) + 1
        has_rows = len(df) > 0 and self.subgroup_field in df.columns
        out = {}
        for v in self.subgroup_values:
            if not totals.get(v):
                continue
            n_correct = int(df.loc[df[self.subgroup_field] == v, "correct"].sum()) if has_rows else 0
            out[v] = round(n_correct / totals[v] * 100, 2)
        return out

    def _print_report(self, accuracy, correct, total, subgroup_acc, latency):
        print(f"\n{'='*50}")
        print(f"Overall: {accuracy}% ({correct}/{total})")
        for v, acc in subgroup_acc.items():
            print(f"  {v:8s}: {acc}%")
        if latency:
            print(f"Selection latency ({latency['granularity']}): "
                  f"mean={latency['per_selection_mean_ms']} ms, "
                  f"p50={latency['per_selection_median_ms']} ms, "
                  f"p95={latency['per_selection_p95_ms']} ms")
        print(f"{'='*50}")
