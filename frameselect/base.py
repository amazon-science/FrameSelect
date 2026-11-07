# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Base class for all frame selection methods."""

from __future__ import annotations

import logging
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from .video_io import VideoInput

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SelectionResult:
    """Output of a frame selection method.

    Attributes:
        indices: Selected frame indices (0-based, in the original video) in temporal order.
        frames: Selected frames as numpy arrays (H, W, 3), uint8.
        scores: Optional relevance/importance scores for each selected frame
            (shape ``(n_selected,)``, aligned with ``indices``).
        all_scores: Optional full per-candidate score vector (shape
            ``(n_candidates,)``), covering every candidate considered
            by the selector — not just the picks. Every selector except
            ``uniform`` fills it; it is ``None`` for ``uniform``. Useful
            for plotting a relevance-over-time curve or for post-hoc
            re-ranking.
        score_kind: Short label describing the semantics of ``scores``
            and ``all_scores``. One of ``"cosine"`` (CLIP / SigLIP
            query-similarity, bounded roughly in ``[-1, 1]``),
            ``"logit"`` (raw CLIP logits, unbounded),
            ``"probability"`` (in ``[0, 1]``), ``"custom"``
            (method-specific, see selector docstring), or ``None`` when
            ``scores`` is ``None``.
        candidate_indices: Optional absolute original-video frame
            indices for every candidate considered (shape
            ``(n_candidates,)``). Aligned with ``all_scores``.
        metadata: Dict with video info, selector provenance, and
            method-specific extras.

            A run through :meth:`BaseSelector.select` populates the
            following standard provenance keys:

            * ``frameselect_version`` — library version string.
            * ``selector`` — registered selector name (e.g. ``"efs"``).
            * ``seed`` — the seed the selector was constructed with
              (``None`` when not set; default-seed behaviour is
              selector-specific, see :class:`BaseSelector`).
            * ``n_frames`` — the ``n_frames`` argument to ``select``.
            * ``candidate_fps`` — the sub-sampling rate that produced
              the candidate pool.
            * ``candidate_frames`` — the cap applied to that pool.
            * ``n_candidates`` — the actual candidate-pool size used
              (== ``len(candidate_indices)``, after the cap).
            * ``native_fps`` — the source video's native frame rate.
            * ``vlm_spec`` — the selector's internal VLM spec string
              (populated for VLM-based selectors; ``None`` otherwise).
            * ``encoder`` / ``encoder_pretrained`` — the frame encoder
              identifier, when applicable.
            * ``query`` — the query string, when provided.
            * ``source`` — a string identifier for the video source
              (path, URL, or in-memory tag).
            * ``cache_hit`` — whether the run used the embedding
              cache fast path.

            Methods may also opt in to the ``debug_traces`` protocol:
            a selector assigns ``self._pending_debug_traces`` while it
            selects, and :meth:`BaseSelector.select` moves it to
            ``metadata["debug_traces"]``. Each trace is itself a dict::

                metadata["debug_traces"] = {
                    "relevance": {
                        "values": np.ndarray,       # shape (n_candidates,)
                        "label": "Relevance",       # y-axis label
                        "kind": "curve",            # or "bar"
                        "higher_is_better": True,   # affects shading
                    },
                    ...
                }

            These are plotted by :func:`frameselect.viz.debug_trajectory`.
            Methods that don't opt in simply fall back to
            :func:`frameselect.viz.score_timeline` on ``all_scores``.
    """

    indices: np.ndarray
    frames: np.ndarray
    scores: Optional[np.ndarray] = None
    all_scores: Optional[np.ndarray] = None
    score_kind: Optional[str] = None
    candidate_indices: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.indices)

    # ------------------------------------------------------------------
    # Serialisation — round-trippable via ``to_dict`` / ``from_dict``.
    # ``frames`` is dropped (binary payload; reload from ``indices``
    # + the original video when needed).
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict representation.

        Drops ``frames`` (binary, reconstructable from ``indices`` +
        the source video). Numpy arrays are serialised as plain lists.
        Metadata is passed through — callers are responsible for its
        JSON-safety (the default provenance keys populated by
        :meth:`BaseSelector.select` already are).

        Returns
        -------
        dict
            Keys: ``indices``, ``scores``, ``all_scores``,
            ``score_kind``, ``candidate_indices``, ``metadata``.
        """
        def _a(x):
            if x is None:
                return None
            return np.asarray(x).tolist()
        return {
            "indices": _a(self.indices),
            "scores": _a(self.scores),
            "all_scores": _a(self.all_scores),
            "score_kind": self.score_kind,
            "candidate_indices": _a(self.candidate_indices),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SelectionResult":
        """Reconstruct a :class:`SelectionResult` from :meth:`to_dict` output.

        ``frames`` is set to an empty ``uint8`` array because the
        serialised form drops it. Callers that need frames should
        re-decode from the source video using ``data["indices"]``.
        """
        def _a(x, dtype=None):
            if x is None:
                return None
            arr = np.asarray(x)
            return arr.astype(dtype) if dtype is not None else arr
        return cls(
            indices=_a(data.get("indices"), dtype=np.int64),
            frames=np.zeros((0, 1, 1, 3), dtype=np.uint8),
            scores=_a(data.get("scores")),
            all_scores=_a(data.get("all_scores")),
            score_kind=data.get("score_kind"),
            candidate_indices=_a(data.get("candidate_indices"), dtype=np.int64),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_json(self) -> str:
        """Return a compact JSON string. Shortcut for ``json.dumps(to_dict())``."""
        import json
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_json(cls, text: str) -> "SelectionResult":
        """Parse a JSON string produced by :meth:`to_json`."""
        import json
        return cls.from_dict(json.loads(text))


def _drain_debug_traces(selector: "BaseSelector", base_meta: dict) -> dict:
    """Fold ``selector._pending_debug_traces`` into ``base_meta``.

    Returns a new dict so we don't mutate the caller's map. Clears the
    scratchpad on the selector so the next ``.select()`` call starts
    fresh. If no traces were written, returns a shallow copy of
    ``base_meta`` with no ``debug_traces`` key.
    """
    out = dict(base_meta)
    traces = getattr(selector, "_pending_debug_traces", None)
    if traces:
        out["debug_traces"] = dict(traces)
        # Reset so the next select() starts clean.
        selector._pending_debug_traces = {}
    return out


def _cached_pool_rows(entry, max_candidates: Optional[int]) -> Optional[np.ndarray]:
    """Rows of a selector-cache entry that form the capped candidate pool.

    Returns the rows of ``entry`` holding exactly the pool an uncached
    :meth:`BaseSelector.select` would build under ``max_candidates``, or
    ``None`` when the entry cannot supply it (it was written under a
    different cap). Entries record the pool size before the cap
    (``meta["n_candidates_uncapped"]``; entries without it hold the whole
    ``candidate_fps`` grid), which fixes the grid positions they store.
    """
    from .video_io import _capped_positions

    n_grid = int(entry.meta.get("n_candidates_uncapped", entry.n_candidates))
    have = _capped_positions(n_grid, entry.n_candidates)
    if len(have) != entry.n_candidates or len(entry.embeddings) != entry.n_candidates:
        return None  # inconsistent entry; re-encode
    want = _capped_positions(n_grid, max_candidates)
    rows = np.searchsorted(have, want)
    if len(rows) and (rows[-1] >= len(have) or not np.array_equal(have[rows], want)):
        return None
    return rows


@dataclass
class _NormalizedOutput:
    """Internal: unified shape for what a selector method returns.

    Selectors return one of:

    - ``None``                                        — embedding path not supported
    - ``(indices, scores)``                           — legacy 2-tuple
    - ``(indices, scores, all_scores)``               — with explainer hook
    - ``(indices, scores, all_scores, score_kind)``   — with score semantics label

    :func:`_normalize_selector_output` folds all four shapes into this
    struct so ``select()`` doesn't need ``if len(out) == N:`` branches.
    """
    indices: np.ndarray
    scores: Optional[np.ndarray] = None
    all_scores: Optional[np.ndarray] = None
    score_kind: Optional[str] = None


def _normalize_selector_output(raw) -> Optional[_NormalizedOutput]:
    """Coerce a selector's raw return into ``_NormalizedOutput`` or ``None``."""
    if raw is None:
        return None
    # Guard the len()-based dispatch below: a bare ndarray (or any other
    # sized non-sequence) of length 2/3/4 would otherwise be silently
    # unpacked as if it were an (indices, scores, ...) tuple. Selectors
    # must return an explicit tuple/list.
    if not isinstance(raw, (tuple, list)):
        raise TypeError(
            f"selector returned {type(raw).__name__}; expected a tuple or "
            "list of (indices, scores[, all_scores[, score_kind]])"
        )
    if len(raw) == 2:
        indices, scores = raw
        return _NormalizedOutput(indices=indices, scores=scores)
    if len(raw) == 3:
        indices, scores, all_scores = raw
        return _NormalizedOutput(
            indices=indices, scores=scores, all_scores=all_scores,
        )
    if len(raw) == 4:
        indices, scores, all_scores, score_kind = raw
        return _NormalizedOutput(
            indices=indices, scores=scores, all_scores=all_scores,
            score_kind=score_kind,
        )
    raise ValueError(
        f"selector returned a {len(raw)}-tuple; expected 2, 3, or 4 "
        "(indices, scores[, all_scores[, score_kind]])"
    )


class BaseSelector(ABC):
    """Abstract base class for frame selectors.

    All selectors implement :meth:`select` which takes a video path
    (and optionally a query) and returns a :class:`SelectionResult`.

    Subclasses must implement :meth:`_select_indices`.

    Determinism
    -----------
    Pass ``seed`` at construction to make stochastic methods
    reproducible::

        sel = fs.create("kmeans", seed=42)
        r1 = sel.select(video, n_frames=8, query=q)
        r2 = sel.select(video, n_frames=8, query=q)
        assert (r1.indices == r2.indices).all()

    ``seed=None`` (the default) keeps each method's original behaviour —
    most methods that use randomness already hardcode an internal seed
    so their outputs are deterministic out of the box. Setting ``seed``
    overrides those internal defaults across every method that uses
    RNG (``kmeans``).

    Note: selector-level determinism does **not** imply VLM-level
    determinism. VLM backends (Bedrock, OpenAI-compatible servers,
    Anthropic API) can produce nondeterministic outputs even at
    ``temperature=0`` due to batching, non-associative float math on
    GPUs, and server-side retries.
    """

    # Whether this method can use a textual query.
    query_aware: bool = False

    # Whether this method reads multiple-choice answer choices. Evaluators
    # then pass the query as the question followed by one choice per line
    # (``"A. ..."``), the format ``frameselect.methods.qav.split_query``
    # parses.
    uses_answer_choices: bool = False

    # Uniform defaults that every selector inherits. Subclasses may
    # override (rare).
    DEFAULT_N_FRAMES: int = 8
    DEFAULT_CANDIDATE_FRAMES: int = 128
    DEFAULT_CANDIDATE_FPS: float = 1.0
    DEFAULT_ENCODER: str = "ViT-B-32"
    DEFAULT_ENCODER_PRETRAINED: str = "openai"

    def __init__(
        self,
        device: str = "cpu",
        seed: Optional[int] = None,
        cache_dir: "str | Path | None" = None,
        *,
        # -- Unified evaluation-axis controls (axes 1-4 of the paper) --
        n_frames: Optional[int] = None,
        candidate_frames: Optional[int] = None,
        candidate_fps: Optional[float] = None,
        encoder: Optional[str] = None,
        encoder_pretrained: Optional[str] = None,
        text_encoder: Optional[str] = None,
        text_encoder_pretrained: Optional[str] = None,
        vlm: Optional[str] = None,
        # -- Legacy aliases (deprecated) --
        clip_model: Optional[str] = None,
        clip_pretrained: Optional[str] = None,
        **kwargs,
    ):
        """Construct a selector.

        Parameters
        ----------
        device : str
            PyTorch device (``"cpu"`` or ``"cuda"``).
        seed : int, optional
            Random seed for stochastic methods. See class docstring.
        cache_dir : str or Path, optional
            Embedding cache directory. If set, ``select()`` reuses
            pre-computed embeddings keyed on (video, encoder, fps).
        n_frames : int, optional
            Default output frame budget. Overridable per call via
            ``select(..., n_frames=...)``. Defaults to
            :attr:`DEFAULT_N_FRAMES`.
        candidate_frames : int, optional
            Maximum candidate pool size per video. When sampling at
            ``candidate_fps`` yields more candidates, ``candidate_frames``
            of them are kept, spread uniformly over the whole video —
            the same rule in :meth:`select` and in the evaluator. ``0``
            disables the cap. Defaults to
            :attr:`DEFAULT_CANDIDATE_FRAMES`.
        candidate_fps : float, optional
            Candidate-pool sampling rate. Defaults to
            :attr:`DEFAULT_CANDIDATE_FPS`.
        encoder : str, optional
            OpenCLIP / SigLIP model name. Ignored by non-CLIP selectors
            (``uniform``).
            Defaults to :attr:`DEFAULT_ENCODER`.
        encoder_pretrained : str, optional
            Pretrained-weights tag (``"openai"``, ``"webli"``, ...).
            Defaults to :attr:`DEFAULT_ENCODER_PRETRAINED`.
        vlm : str, optional
            Selection-VLM spec (e.g. ``"openai:Qwen/Qwen2.5-VL-7B-Instruct"``).
            Used only by selectors that call a VLM during selection
            through :mod:`frameselect.vlm.client` (none of the bundled
            selectors does). Stored on the selector; no global side-effects. An explicit
            but unknown backend prefix raises ``ValueError`` here rather
            than silently falling back to another backend.
        clip_model, clip_pretrained : str, optional
            Legacy aliases for ``encoder`` / ``encoder_pretrained``.
            Accepted for backward compatibility; prefer the new names.
        """
        self.device = device
        self.seed = seed

        # -- Evaluation axes live on the selector --
        # These are per-selector attributes so two selectors in the
        # same process can be configured differently.
        self.n_frames = n_frames if n_frames is not None else self.DEFAULT_N_FRAMES
        self.candidate_frames = (
            candidate_frames if candidate_frames is not None
            else self.DEFAULT_CANDIDATE_FRAMES
        )
        self.candidate_fps = (
            candidate_fps if candidate_fps is not None
            else self.DEFAULT_CANDIDATE_FPS
        )
        # Resolve encoder: ``encoder`` wins, then legacy ``clip_model``,
        # then the selector's class-level :attr:`DEFAULT_ENCODER`.
        if encoder is not None:
            resolved_encoder = encoder
        elif clip_model is not None:
            resolved_encoder = clip_model
        else:
            resolved_encoder = self.DEFAULT_ENCODER
        if encoder_pretrained is not None:
            resolved_pretrained = encoder_pretrained
        elif clip_pretrained is not None:
            resolved_pretrained = clip_pretrained
        else:
            resolved_pretrained = self.DEFAULT_ENCODER_PRETRAINED
        self.encoder = resolved_encoder
        self.encoder_pretrained = resolved_pretrained

        # -- Optional decoupled TEXT encoder (multilingual experiments) --
        # When set, query text is encoded by a different tower than the
        # frame (image) tower, while ``self.encoder`` stays the IMAGE tower
        # so the frame-embedding cache key is unchanged (shared with the
        # English-CLIP baseline). ``None`` (default) = frames and text share
        # one CLIPEncoder, exactly as before. See ``_construct_encoder``.
        self.text_encoder = text_encoder
        self.text_encoder_pretrained = text_encoder_pretrained

        # Back-compat shims so existing selector subclasses that read
        # ``self.clip_model`` / ``self.clip_pretrained`` keep working.
        # These stay in sync with ``self.encoder`` / ``self.encoder_pretrained``.
        self.clip_model = self.encoder
        self.clip_pretrained = self.encoder_pretrained

        # -- VLM spec is selector-local (no global singleton). --
        # A selector that calls a VLM reads ``self._vlm_spec`` and routes
        # its calls through ``vlm.client.call_text`` / ``call_vision`` with
        # the spec passed explicitly. Validate it here, so a typo'd backend
        # prefix fails at construction rather than at call time.
        if isinstance(vlm, str):
            from .vlm.client import _parse_vlm_spec
            _parse_vlm_spec(vlm)
        self._vlm_spec = vlm

        # Managed vLLM server — started by warmup() if needed, stopped
        # when the selector is garbage-collected.
        self._managed_server = None
        # Scratchpad for method-contributed debug traces (see viz.debug_trajectory).
        # Methods opt in by assigning a ``dict[str, dict]`` to
        # ``self._pending_debug_traces`` during ``_select_indices`` or
        # ``_select_from_embeddings``; ``select()`` drains it into
        # ``SelectionResult.metadata["debug_traces"]`` and resets the
        # attribute. Methods that don't write here simply get no trace.
        #
        # Backed by thread-local storage: ``select_batch`` shares one
        # selector instance across worker threads, and each worker's
        # ``select()`` writes then drains its own traces. Without
        # per-thread isolation, thread A could drain the traces thread B
        # just wrote. The ``_pending_debug_traces`` property (below)
        # routes reads/writes to the calling thread's slot, so methods
        # keep assigning to ``self._pending_debug_traces`` unchanged.
        self._debug_trace_local = threading.local()
        # Guards the lazy build in the default ``_get_encoder`` so
        # concurrent first calls from ``select_batch`` workers build one
        # encoder, not one per thread.
        self._default_encoder_lock = threading.Lock()
        # Optional embedding cache. When set, ``select()`` will
        # look up (video, encoder, candidate_fps) and skip CLIP
        # encoding on cache hits.
        #
        # Resolution order:
        #   1. explicit ``cache_dir`` kwarg wins
        #   2. ``FRAMESELECT_CACHE_DIR`` env var activates the cache at
        #      its XDG default path
        #   3. no cache (default) — same behaviour as before
        from .cache import EmbeddingCache
        self._cache: Optional[EmbeddingCache] = None
        if cache_dir is not None:
            self._cache = EmbeddingCache(cache_dir)
        elif os.environ.get("FRAMESELECT_CACHE_DIR"):
            self._cache = EmbeddingCache()  # picks up the env var

    @property
    def _pending_debug_traces(self) -> dict:
        """Per-thread debug-trace scratchpad (see ``__init__``).

        Thread-local so concurrent ``select()`` calls in
        :meth:`select_batch` don't clobber each other's traces.
        """
        return getattr(self._debug_trace_local, "traces", {})

    @_pending_debug_traces.setter
    def _pending_debug_traces(self, value: dict) -> None:
        self._debug_trace_local.traces = value

    def _construct_encoder(self):
        """Build this selector's vision-language encoder.

        The default is a single :class:`~frameselect.encoders.CLIPEncoder`
        whose one OpenCLIP model serves BOTH the image and text towers —
        the historical behaviour.

        When ``self.text_encoder`` is set to ``"mclip:<hf-id-suffix>"`` (e.g.
        ``"mclip:XLM-Roberta-Large-Vit-B-32"``), a
        :class:`~frameselect.encoders.MultilingualCLIPEncoder` is returned:
        it keeps ``self.encoder`` / ``self.encoder_pretrained`` as the frozen
        IMAGE tower (so ``encode_frames`` — and therefore the frame-embedding
        cache — is byte-identical to the CLIP baseline) but routes
        ``encode_text`` / ``similarity`` through the M-CLIP multilingual text
        tower. This isolates the text encoder as the only changed variable in
        a multilingual selection experiment.

        Selectors that build a CLIP encoder should call this instead of
        constructing ``CLIPEncoder`` directly, so the ``text_encoder`` axis
        takes effect uniformly.
        """
        from .encoders import CLIPEncoder

        text_spec = getattr(self, "text_encoder", None)
        if text_spec and str(text_spec).startswith("mclip:"):
            from .encoders import MultilingualCLIPEncoder

            text_id = "M-CLIP/" + str(text_spec).split(":", 1)[1]
            return MultilingualCLIPEncoder(
                text_model=text_id,
                image_model=self.encoder,
                image_pretrained=self.encoder_pretrained,
                device=self.device,
            )
        return CLIPEncoder(self.encoder, self.encoder_pretrained, self.device)

    def _get_encoder(self):
        """Return this selector's encoder, building it on first use.

        The default builds :meth:`_construct_encoder` once and caches it
        on ``self._encoder``, so a custom selector can call
        ``self._get_encoder()`` without defining it. The library's
        embedding-based selectors (kmeans, efs, agfs, qav) override
        this method; only such an override marks a selector as
        encoder-backed for the embedding cache and :meth:`warmup` (see
        :meth:`_uses_own_encoder`).
        """
        encoder = getattr(self, "_encoder", None)
        if encoder is None:
            with self._default_encoder_lock:
                encoder = getattr(self, "_encoder", None)
                if encoder is None:
                    encoder = self._construct_encoder()
                    self._encoder = encoder
        return encoder

    def _uses_own_encoder(self) -> bool:
        """Whether this selector supplies its own ``_get_encoder``.

        True for a subclass override or an instance-level assignment
        (``sel._get_encoder = ...``); False when only the
        :class:`BaseSelector` default is present — as for the rule-based
        ``uniform`` selector. Those never trigger embedding caching or
        encoder preloading.
        """
        if "_get_encoder" in vars(self):
            return True
        return type(self)._get_encoder is not BaseSelector._get_encoder

    def _seeded_rng(self, default: int = 42) -> np.random.Generator:
        """Return a numpy ``Generator`` seeded from ``self.seed``.

        Falls back to ``default`` when the user didn't specify a seed.
        Methods should call this to get per-call RNGs — this keeps
        them forward-compatible if we ever need to offset seeds
        (e.g., for ``select_batch``).
        """
        seed = getattr(self, "seed", None)
        s = seed if seed is not None else default
        return np.random.default_rng(s)

    def _seed_legacy(self, default: int) -> int:
        """Return an int seed for APIs that want ``int`` (sklearn, legacy RandomState).

        ``default`` preserves the method's historical hardcoded value
        so reproductions remain byte-identical when the user doesn't
        pass a seed.
        """
        seed = getattr(self, "seed", None)
        return seed if seed is not None else default

    def _hyperparams(self) -> dict:
        """Return this selector's method-specific hyperparameters.

        Subclasses override to expose their tunable knobs (e.g.
        :attr:`EFSSelector.max_events`, :attr:`AGFSSelector.alpha`) so they
        appear in :attr:`SelectionResult.metadata` under the
        ``hyperparams`` key. The base implementation returns an empty
        dict, so selectors without such knobs (kmeans) stay silent.

        The purpose is provenance, not reproducibility: the universal
        axes (``seed``, ``n_frames``, ``encoder``, ``vlm``, ...) are
        already captured by :meth:`_build_provenance`. This hook only
        covers the method-specific tail (``max_events``, ``preset``,
        ``rho``, etc.) that hyperparameter sweeps vary.

        Example
        -------
        An EFS ``max_events`` sweep is self-documenting in the result JSON::

            for m in [8, 10, 12]:
                sel = fs.create("efs", max_events=m, n_frames=16)
                r = ev.run(sel)
                print(r.df.iloc[0].selected_indices,
                      "max_events =", r.provenance["hyperparams"].get("max_events"))

        Returns
        -------
        dict
            A JSON-serialisable mapping of hyperparameter name → value.
        """
        return {}

    def warmup(self) -> None:
        """Pre-load the encoder and start any required VLM server.

        Called automatically by the evaluator before the parallel
        selection loop. Override in subclasses that need custom warmup
        (e.g. loading a large model). The base implementation handles
        the common case: if ``vlm="openai:..."`` was passed and no
        server is reachable at the resolved base URL, auto-starts a
        local vLLM server on the resolved port using all available
        GPUs. The server is stopped when the selector is garbage
        collected (see :meth:`__del__`).

        It also preloads the frame encoder (when the selector has one)
        so the potentially expensive model build happens once, in the
        main thread, before :meth:`select_batch` fans out across worker
        threads. This upholds the documented ``select_batch`` invariant
        and avoids concurrent ``_get_encoder`` calls racing on lazy
        first-use init.
        """
        # Preload the encoder in the calling thread (main thread when
        # invoked from select_batch), so worker threads all reuse the
        # already-built encoder instead of racing to construct it.
        if self._uses_own_encoder():
            try:
                self._get_encoder()
            except Exception:
                # A failed encoder build shouldn't block warmup; the
                # real error re-surfaces per-video during selection.
                logger.warning(
                    "warmup: encoder preload failed for %s",
                    type(self).__name__,
                    exc_info=True,
                )

        if not self._vlm_spec or not self._vlm_spec.startswith("openai:"):
            return

        from .vlm.openai import _probe_local_vllm, _default_base_url
        # Resolve the server URL / port the same way the openai client
        # does: env var if set, else auto-probe ``localhost:8000``.
        base_url = _default_base_url() or "http://localhost:8000/v1"
        from urllib.parse import urlparse
        p = urlparse(base_url)
        host, port = p.hostname or "localhost", p.port or 8000

        if _probe_local_vllm(host=host, port=port) is not None:
            return  # already reachable

        model = self._vlm_spec.split(":", 1)[1]
        from .vlm.server import VLMServer
        self._managed_server = VLMServer.start(
            model=model,
            port=port,
        )

    def __del__(self):
        """Stop any managed vLLM server when the selector is garbage-collected."""
        server = getattr(self, "_managed_server", None)
        if server is not None:
            try:
                server.stop(verbose=False)
            except Exception:
                pass

    @property
    def method_name(self) -> str:
        """Registry key for this selector (e.g. ``"efs"``, ``"uniform"``).

        Looks up the class in the registry and returns its registered name.
        Falls back to the lowercased class name if not found.
        """
        from .registry import _REGISTRY
        for name, cls in _REGISTRY.items():
            if cls is type(self):
                return name
        return type(self).__name__.lower().replace("selector", "")

    @abstractmethod
    def _select_indices(
        self,
        frames: np.ndarray,
        n_select: int,
        query: Optional[str] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Return ``(selected_indices, optional_scores)`` or
        ``(selected_indices, optional_scores, optional_all_scores)``.

        Parameters
        ----------
        frames : np.ndarray
            Candidate frames, shape (T, H, W, 3), uint8.
            These are already pre-sampled at ``candidate_fps``.
        n_select : int
            Number of frames to select.
        query : str, optional
            Textual query (only used by query-aware methods).

        Returns
        -------
        indices : np.ndarray of int
            Indices into *frames* (not original video indices).
        scores : np.ndarray or None
            Per-selected-frame scores (higher = more relevant),
            aligned with ``indices``.
        all_scores : np.ndarray or None, optional third element
            Full per-candidate score vector of shape ``(T,)``, covering
            every candidate considered. Used by the explainer API
            (``SelectionResult.all_scores``). Methods that don't
            naturally produce a dense score vector omit this element.
        """
        ...

    def _select_from_embeddings(
        self,
        embeddings: np.ndarray,
        n_select: int,
        query: Optional[str] = None,
        query_embedding: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Fast path: select using pre-computed CLIP embeddings.

        Override in subclasses that support this. Default returns None
        to signal the caller should fall back to ``_select_indices``.

        Parameters
        ----------
        embeddings : np.ndarray, shape (T, D)
            Pre-computed L2-normalized CLIP frame embeddings.
        n_select : int
            Number of frames to select.
        query : str, optional
            Textual query (used to encode if query_embedding not provided).
        query_embedding : np.ndarray, optional
            Pre-computed query text embedding. If provided, skips CLIP text encoding.

        Returns
        -------
        (indices, scores) or None if not supported.
        """
        return None

    def _cache_config(self) -> Optional[dict]:
        """Describe the encoder config used for caching, or ``None``.

        Subclasses that expose a fast ``_select_from_embeddings`` path
        should return a dict identifying the encoder they use, so
        :meth:`select` can key the embedding cache on it. Methods
        without an embedding fast path (uniform) return
        ``None`` and never hit the cache.

        Expected keys::

            {"encoder": "<model_name>",
             "pretrained": "<weights_tag>"}
        """
        # An encoder is only meaningful for selectors that actually use
        # one. We detect that by a ``_get_encoder`` override on the
        # subclass — the uniform/value-free factory pattern shared across
        # all CLIP-based selectors. Pure rule-based selectors (uniform)
        # only inherit the base default and therefore have no
        # cacheable encoder footprint.
        if not self._uses_own_encoder():
            return None
        model = getattr(self, "encoder", None)
        pre = getattr(self, "encoder_pretrained", None)
        if model is None or pre is None:
            return None
        return {"encoder": model, "pretrained": pre}

    def select(
        self,
        video: "VideoInput",
        n_frames: Optional[int] = None,
        query: Optional[str] = None,
        *,
        candidate_fps: Optional[float] = None,
    ) -> SelectionResult:
        """Select frames from a video.

        The video is first pre-sampled into candidate frames
        (default 1 FPS), capped at ``self.candidate_frames`` candidates
        spread uniformly over the whole video (default 128; the same
        pool the evaluator builds), then the selection algorithm picks
        ``n_frames`` from those candidates. This two-stage approach
        keeps things fast even for long videos.

        This method is **task-agnostic** — it returns frames and
        scores. What you do with them is up to you: pass them to a
        VLM for question answering, a captioning model, a retrieval
        pipeline, a classifier, a notebook display, anything.

        Parameters
        ----------
        video : str, Path, np.ndarray, torch.Tensor, list[PIL.Image], or decord.VideoReader
            The video to select frames from. Supported inputs:

            * ``str`` / ``pathlib.Path`` — local path, or a URL
              (``http://``, ``https://``, ``s3://``, ``gs://``, ``ftp://``).
            * ``np.ndarray`` of shape ``(T, H, W, 3)`` uint8.
            * ``torch.Tensor`` of shape ``(T, C, H, W)`` or ``(T, H, W, C)``.
            * ``list[PIL.Image.Image]``.
            * A pre-opened ``decord.VideoReader``.

            See :func:`frameselect.video_io.load_video_frames` for details.
        n_frames : int, optional
            Number of frames to select. ``None`` (the default) uses
            ``self.n_frames`` — the budget configured at construction
            (:attr:`DEFAULT_N_FRAMES` unless overridden). Pass a value
            here to override on a per-call basis.
        query : str, optional
            Textual query for query-aware methods.
        candidate_fps : float, optional
            Pre-sampling rate for candidate frames. ``None`` uses
            ``self.candidate_fps`` (1 FPS by default). ``0`` samples
            every frame. Any positive float samples at that rate. The
            ``candidate_frames`` cap applies either way.

        Returns
        -------
        SelectionResult
        """
        # Resolve defaults from the selector's configuration.
        if n_frames is None:
            n_frames = self.n_frames
        if candidate_fps is None:
            candidate_fps = self.candidate_fps

        # Three execution paths, dispatched linearly:
        #
        #   1. cache-hit        → skip video decode + encoder forward
        #   2. cache-populate   → decode + encode + write cache + pick
        #   3. no-cache         → decode + pick on raw frames
        #
        # Each path returns a ``SelectionResult`` or ``None``. ``None``
        # means "this path doesn't apply, try the next one".
        cache_key = self._resolve_cache_key(video, candidate_fps)

        if cache_key is not None:
            result = self._select_via_cache_hit(
                video, cache_key, n_frames, query, candidate_fps
            )
            if result is not None:
                return result

        return self._select_via_full_path(
            video, cache_key, n_frames, query, candidate_fps
        )

    # ------------------------------------------------------------------
    # select() — internal helpers
    # ------------------------------------------------------------------

    def _resolve_cache_key(
        self, video: "VideoInput", candidate_fps: Optional[float]
    ) -> Optional[str]:
        """Return the cache key for this (video, encoder, fps), or ``None``.

        ``None`` means caching isn't applicable — either no cache is
        configured, the selector has no encoder fast path, or the
        video is an in-memory input we can't stably hash.
        """
        cache = getattr(self, "_cache", None)
        if cache is None:
            return None
        cache_cfg = self._cache_config()
        if cache_cfg is None:
            return None
        from .cache import make_key
        cfps = 1.0 if candidate_fps is None else float(candidate_fps)
        return make_key(
            video,
            encoder_name=cache_cfg["encoder"],
            encoder_pretrained=cache_cfg["pretrained"],
            candidate_fps=cfps,
        )

    def _select_via_cache_hit(
        self,
        video: "VideoInput",
        cache_key: str,
        n_frames: int,
        query: Optional[str],
        candidate_fps: Optional[float],
    ) -> Optional[SelectionResult]:
        """Fast path: run the embedding selector against cached tensors.

        Returns ``None`` when the cache has no entry for ``cache_key``
        or the selector declines the embedding path.
        """
        entry = self._cache.get(cache_key)
        if entry is None:
            return None
        # The key leaves out ``candidate_frames``, so serve the entry only
        # if it holds the pool an uncached call would build under this
        # selector's cap; otherwise it is a miss and gets re-populated.
        rows = _cached_pool_rows(entry, self.candidate_frames)
        if rows is None:
            return None
        embeddings = entry.embeddings[rows]
        candidate_indices = entry.candidate_indices[rows]

        n_select = min(n_frames, len(rows))
        out = _normalize_selector_output(
            self._select_from_embeddings(
                embeddings, n_select, query, query_embedding=None,
            )
        )
        if out is None:
            return None

        if entry.frames_path is not None:
            from .cache import _load_frames_bin
            try:
                all_frames_full = _load_frames_bin(entry.frames_path)
                if len(all_frames_full) != entry.n_candidates:
                    # Entries under one key can hold different pools (the
                    # key leaves out the cap), so a sidecar left by another
                    # write would silently index the wrong frames.
                    raise ValueError(
                        f"sidecar holds {len(all_frames_full)} frames, "
                        f"entry has {entry.n_candidates} candidates"
                    )
                picked = all_frames_full[rows[out.indices]]
            except Exception:
                # A truncated / corrupt frames sidecar (or an out-of-range
                # index into it) must not crash select(). Treat it as a
                # cache miss: returning None lets select() fall through to
                # the full decode+re-populate path, which overwrites the
                # bad sidecar.
                logger.warning(
                    "cache-hit: failed to read frames sidecar %s; treating "
                    "as a cache miss",
                    entry.frames_path,
                    exc_info=True,
                )
                return None
        else:
            from .video_io import load_video_frames
            pool_frames, _, _ = load_video_frames(
                video, candidate_fps=candidate_fps,
                max_candidates=self.candidate_frames,
            )
            picked = pool_frames[out.indices]

        abs_indices = candidate_indices[out.indices]
        return SelectionResult(
            indices=abs_indices,
            frames=picked,
            scores=out.scores,
            all_scores=out.all_scores,
            score_kind=out.score_kind,
            candidate_indices=candidate_indices,
            metadata=_drain_debug_traces(
                self,
                self._build_provenance(
                    n_frames=n_frames,
                    candidate_fps=candidate_fps,
                    query=query,
                    video_info=entry.meta,
                    candidate_indices=candidate_indices,
                    cache_hit=True,
                ),
            ),
        )

    def _select_via_full_path(
        self,
        video: "VideoInput",
        cache_key: Optional[str],
        n_frames: int,
        query: Optional[str],
        candidate_fps: Optional[float],
    ) -> SelectionResult:
        """Slow path: decode the video, optionally populate the cache,
        then pick frames. Always returns a result (never ``None``).
        """
        from .video_io import load_video_frames
        # Same candidate pool as the evaluator: the ``candidate_fps``
        # grid, capped at ``candidate_frames`` uniformly over the video.
        all_frames, original_indices, video_info = load_video_frames(
            video, candidate_fps=candidate_fps,
            max_candidates=self.candidate_frames,
        )

        # If caching is configured, try the encoder fast path:
        # encode once, write to disk, pick from embeddings.
        if cache_key is not None:
            result = self._try_cache_populate_and_select(
                cache_key, all_frames, original_indices, video_info,
                n_frames, query, candidate_fps,
            )
            if result is not None:
                return result

        # Last resort — method's pixel-space selector on raw frames.
        n_select = min(n_frames, len(all_frames))
        out = _normalize_selector_output(
            self._select_indices(all_frames, n_select, query)
        )
        assert out is not None, "_select_indices must return indices"
        abs_indices = original_indices[out.indices]
        return SelectionResult(
            indices=abs_indices,
            frames=all_frames[out.indices],
            scores=out.scores,
            all_scores=out.all_scores,
            score_kind=out.score_kind,
            candidate_indices=original_indices,
            metadata=_drain_debug_traces(
                self,
                self._build_provenance(
                    n_frames=n_frames,
                    candidate_fps=candidate_fps,
                    query=query,
                    video_info=video_info,
                    candidate_indices=original_indices,
                    cache_hit=False,
                ),
            ),
        )

    def _try_cache_populate_and_select(
        self,
        cache_key: str,
        all_frames: np.ndarray,
        original_indices: np.ndarray,
        video_info: dict,
        n_frames: int,
        query: Optional[str],
        candidate_fps: Optional[float],
    ) -> Optional[SelectionResult]:
        """Encode frames once, write the cache, and pick via the
        embedding fast path. Returns ``None`` if this selector has no
        encoder, encoding fails, or the embedding path declines to
        run — in which case the caller falls through to raw-pixel
        selection.
        """
        encoder = self._get_cache_encoder()
        if encoder is None:
            return None
        try:
            embeddings = encoder.encode_frames(all_frames)
        except Exception:
            # Encoding can fail for real reasons (CUDA OOM, a genuine bug
            # in the encoder). Log before falling back to the raw-pixel
            # path so the failure isn't silently swallowed.
            logger.warning(
                "cache-populate: encode_frames failed for %s; falling back "
                "to raw-pixel selection",
                type(self).__name__,
                exc_info=True,
            )
            return None
        if embeddings is None:
            return None

        cache_cfg = self._cache_config()
        self._cache.put(
            cache_key,
            candidate_indices=original_indices,
            embeddings=embeddings,
            frames=all_frames,
            meta={
                "encoder": cache_cfg["encoder"] if cache_cfg else None,
                "pretrained": cache_cfg["pretrained"] if cache_cfg else None,
                "source": video_info.get("source"),
                # Pool size before the ``candidate_frames`` cap: pins down
                # which grid positions this entry holds, so a later read
                # under another cap can tell whether it is reusable.
                "n_candidates_uncapped": int(video_info.get(
                    "n_candidates_uncapped", len(original_indices),
                )),
                # Timing, so results served from the cache keep a real
                # time axis in the plots.
                **{k: video_info.get(k) for k in (
                    "total_frames", "native_fps", "duration_sec", "candidate_fps_used",
                )},
            },
        )

        n_select = min(n_frames, len(all_frames))
        out = _normalize_selector_output(
            self._select_from_embeddings(
                embeddings, n_select, query, query_embedding=None,
            )
        )
        if out is None:
            return None

        abs_indices = original_indices[out.indices]
        return SelectionResult(
            indices=abs_indices,
            frames=all_frames[out.indices],
            scores=out.scores,
            all_scores=out.all_scores,
            score_kind=out.score_kind,
            candidate_indices=original_indices,
            metadata=_drain_debug_traces(
                self,
                self._build_provenance(
                    n_frames=n_frames,
                    candidate_fps=candidate_fps,
                    query=query,
                    video_info=video_info,
                    candidate_indices=original_indices,
                    cache_hit=False,
                ),
            ),
        )

    def _get_cache_encoder(self):
        """Return the selector's encoder object for cache population,
        or ``None``. We only accept objects with an ``encode_frames``
        method — everything else falls back to the raw-pixel path.
        """
        if not self._uses_own_encoder():
            return None
        try:
            encoder = self._get_encoder()
        except Exception:
            # Building the encoder can fail for real reasons (CUDA OOM,
            # a missing weights download, a genuine bug). Log before
            # falling back to the raw-pixel path.
            logger.warning(
                "cache-populate: _get_encoder failed for %s; falling back "
                "to raw-pixel selection",
                type(self).__name__,
                exc_info=True,
            )
            return None
        return encoder if hasattr(encoder, "encode_frames") else None

    def _build_provenance(
        self,
        *,
        n_frames: int,
        candidate_fps: Optional[float],
        query: Optional[str],
        video_info: dict,
        candidate_indices: np.ndarray,
        cache_hit: bool,
    ) -> dict:
        """Return a dict of standard provenance keys for the result
        ``metadata``. See :class:`SelectionResult` for the contract.

        ``video_info`` is the loader-provided dict (``native_fps``,
        ``source``, ``candidate_fps_used``) and is merged first so
        per-video keys stay in place; provenance keys are added on
        top and always win on conflict.
        """
        from . import __version__ as _fs_version

        cache_cfg = self._cache_config()
        if cache_cfg is None and self._uses_own_encoder():
            # An encoder-backed selector without an embedding cache (it reads
            # frames and encodes them itself) still records which encoder.
            cache_cfg = {"encoder": getattr(self, "encoder", None),
                         "pretrained": getattr(self, "encoder_pretrained", None)}
        prov = dict(video_info)
        prov.update({
            "frameselect_version": _fs_version,
            "selector": self.method_name,
            "seed": self.seed,
            "n_frames": int(n_frames),
            "candidate_fps": (
                float(candidate_fps) if candidate_fps is not None
                else video_info.get("candidate_fps_used")
            ),
            "candidate_frames": self.candidate_frames,
            "n_candidates": int(len(candidate_indices)),
            "vlm_spec": getattr(self, "_vlm_spec", None),
            "encoder": (cache_cfg or {}).get("encoder"),
            "encoder_pretrained": (cache_cfg or {}).get("pretrained"),
            "query": query,
            "cache_hit": cache_hit,
            "hyperparams": dict(self._hyperparams()),
        })
        return prov

    def select_batch(
        self,
        videos,
        n_frames: Optional[int] = None,
        queries=None,
        *,
        candidate_fps: Optional[float] = None,
        n_workers: Optional[int] = None,
        show_progress: bool = True,
        on_error: str = "record",
    ) -> list[SelectionResult]:
        """Select frames from many videos in parallel.

        Runs ``self.select(video, n_frames, query)`` concurrently on
        the inputs, sharing this selector instance across worker
        threads. The encoder (if any) is loaded once in the main
        thread via :meth:`warmup` before workers start, so every
        subsequent call hits the warm cache rather than racing on
        first-use lazy init.

        Parameters
        ----------
        videos : Iterable
            Iterable of video inputs — same accepted types as
            :meth:`select` (paths, URLs, numpy, tensors, PIL lists,
            decord readers). Consumed greedily into a list.
        n_frames : int
            Number of frames to pick per video.
        queries : str or list[str], optional
            Textual queries for query-aware methods.

            * ``None`` (default) — allowed for query-agnostic methods;
              raises for query-aware ones.
            * ``str`` — same query reused for every video.
            * ``list[str]`` — one query per video; must match ``len(videos)``.
        candidate_fps : float, optional
            Forwarded to each :meth:`select` call.
        n_workers : int, optional
            Parallelism. Defaults to
            ``min(32, len(videos), os.cpu_count() or 1)``. Set to 1 to
            run serially (useful for debugging).
        show_progress : bool
            Show a tqdm progress bar. Default ``True``.
        on_error : {"record", "raise", "skip"}
            How to handle per-video errors:

            * ``"record"`` (default) — store the exception on the
              returned result's ``metadata["error"]`` and keep going.
              Returned in input order with a zero-length
              :class:`SelectionResult`.
            * ``"raise"`` — propagate the first exception immediately.
            * ``"skip"`` — drop failed videos from the output. The
              returned list may be shorter than ``videos``.

        Returns
        -------
        list[SelectionResult]
            One result per input video, in input order (unless
            ``on_error="skip"``).

        Examples
        --------
        Batch-select on a directory of clips::

            from pathlib import Path
            import frameselect as fs

            sel = fs.create("efs")
            clips = sorted(Path("clips").glob("*.mp4"))
            results = sel.select_batch(clips, n_frames=16,
                                       queries="what's happening?")
            for clip, r in zip(clips, results):
                print(clip.name, r.indices.tolist())
        """
        import os
        from concurrent.futures import ThreadPoolExecutor, as_completed

        videos = list(videos)
        n = len(videos)

        # Normalize queries -------------------------------------------
        if isinstance(queries, str):
            queries = [queries] * n
        elif queries is None:
            if self.query_aware:
                raise ValueError(
                    f"{type(self).__name__} is query-aware; pass queries= "
                    "(a str or a list[str] the same length as videos)."
                )
            queries = [None] * n
        else:
            queries = list(queries)
            if len(queries) != n:
                raise ValueError(
                    f"len(queries) = {len(queries)} != len(videos) = {n}"
                )

        if on_error not in {"record", "raise", "skip"}:
            raise ValueError(
                f"on_error must be 'record', 'raise', or 'skip', got {on_error!r}"
            )

        if n == 0:
            return []

        if n_workers is None:
            n_workers = min(32, n, os.cpu_count() or 1)
        n_workers = max(1, n_workers)

        # Warm the encoder / backend exactly once before threads start,
        # so we don't pay the download / model-load latency on every
        # worker's first call.
        if hasattr(self, "warmup"):
            try:
                self.warmup()
            except Exception:
                # warmup failures shouldn't block the batch; the error
                # will re-surface per-video if it was real.
                pass

        def _one(i):
            try:
                r = self.select(
                    videos[i], n_frames=n_frames,
                    query=queries[i], candidate_fps=candidate_fps,
                )
                return i, r, None
            except Exception as exc:
                return i, None, exc

        # Serial fast path so users can step through in a debugger.
        if n_workers == 1:
            iterator = (_one(i) for i in range(n))
        else:
            ex = ThreadPoolExecutor(max_workers=n_workers)
            futures = [ex.submit(_one, i) for i in range(n)]
            iterator = (f.result() for f in as_completed(futures))

        # Collect; apply progress bar.
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, total=n, desc="select_batch")
            except ImportError:
                pass

        out: list[tuple[int, Optional[SelectionResult], Optional[BaseException]]] = []
        try:
            for triple in iterator:
                out.append(triple)
        finally:
            if n_workers > 1:
                ex.shutdown(wait=True)

        # Apply on_error policy.
        # Results may be out of order (as_completed) — re-sort by input idx.
        out.sort(key=lambda t: t[0])
        results: list[SelectionResult] = []
        for i, r, err in out:
            if err is None:
                results.append(r)
                continue
            if on_error == "raise":
                raise err
            if on_error == "skip":
                continue
            # "record"
            empty = SelectionResult(
                indices=np.zeros(0, dtype=np.int64),
                frames=np.zeros((0, 1, 1, 3), dtype=np.uint8),
                metadata={
                    "error": repr(err),
                    "source": str(videos[i])[:200],
                },
            )
            results.append(empty)
        return results
