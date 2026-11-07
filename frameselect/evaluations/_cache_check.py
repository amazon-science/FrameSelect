# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Check an evaluator cache file against the selector about to use it.

The evaluator's embedding cache is one ``<cache_dir>/<video>.npz`` per
video, so the *directory*, not the file name, fixes the encoder and the
candidate pool. Without a check, a selector configured with another
encoder, ``candidate_frames`` or ``candidate_fps`` silently selects from
incompatible embeddings (a query-aware one at least crashes on the matmul;
a query-agnostic one does not). :func:`check_cache_matches_selector` compares
whatever a cache file records about how it was built with the selector and
raises :class:`CacheMismatchError` on a mismatch.

Two on-disk layouts exist:

* :class:`~frameselect.cache.EmbeddingCache` entries (written by the
  evaluator's own precompute) keep a pickled ``meta`` dict with
  ``encoder``, ``pretrained``, ``candidate_fps`` and ``native_fps``.
* Caches written by other precompute code may store ``candidate_fps``,
  ``native_fps`` and ``total_frames`` as plain arrays and record no
  encoder. For those only the pool and the embedding width can be checked,
  and a warning says so once per directory and encoder.
"""

from __future__ import annotations

import logging
import os
import pickle
import zipfile
from functools import lru_cache
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class CacheMismatchError(ValueError):
    """An evaluator cache file was built for another encoder or candidate pool."""


# ---------------------------------------------------------------------------
# Reading what a cache file records
# ---------------------------------------------------------------------------

# Globals an ``EmbeddingCache`` ``meta`` entry needs to unpickle: it is a 0-d
# object ndarray holding a plain dict. Anything else -- the usual pickle
# code-execution vector -- is refused, so reading it never needs
# ``allow_pickle=True``.
_META_PICKLE_GLOBALS = frozenset({
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
})

# Plain-array fields of the older layout (also harmless to read from the
# newer one, which does not have them).
_ARRAY_META_KEYS = ("candidate_fps", "native_fps", "total_frames")

# Meta values that record nothing (``_auto_precompute_cache`` writes
# "unknown" when the encoder object has no ``model_name``).
_UNRECORDED = (None, "", "unknown")


class _MetaUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _META_PICKLE_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"cache meta references {module}.{name}; refusing to load it"
        )


def _read_pickled_meta(cache_path: str) -> dict:
    """Return the ``meta`` dict of an ``EmbeddingCache`` npz, or ``{}``."""
    with zipfile.ZipFile(cache_path) as zf:
        if "meta.npy" not in zf.namelist():
            return {}
        with zf.open("meta.npy") as fp:
            version = np.lib.format.read_magic(fp)
            if version == (1, 0):
                np.lib.format.read_array_header_1_0(fp)
            elif version == (2, 0):
                np.lib.format.read_array_header_2_0(fp)
            else:
                return {}
            meta = _MetaUnpickler(fp).load()
    if isinstance(meta, np.ndarray) and meta.shape == ():
        meta = meta.item()
    return dict(meta) if isinstance(meta, dict) else {}


def read_cache_meta(npz, cache_path: str) -> dict:
    """What the cache file ``npz`` (an open ``np.load`` result) records about
    how it was built: any of ``encoder``, ``pretrained``, ``candidate_fps``,
    ``native_fps``, ``total_frames``. Empty when it records nothing.
    """
    meta = {}
    for key in _ARRAY_META_KEYS:
        if key in npz.files:
            try:
                meta[key] = npz[key].item()
            except (TypeError, ValueError):
                pass
    if "meta" in npz.files:
        try:
            meta.update(_read_pickled_meta(cache_path))
        except Exception as exc:
            logger.debug("could not read cache meta from %s: %s: %s",
                         cache_path, type(exc).__name__, exc)
    return meta


# ---------------------------------------------------------------------------
# Comparing a cache file with a selector
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def encoder_output_dim(encoder_name: Optional[str]) -> Optional[int]:
    """Embedding width of a built-in OpenCLIP model, or ``None`` if unknown.

    Reads the model config only -- no weights, no network. Names with a
    ``hf-hub:`` / ``local-dir:`` schema would need I/O and return ``None``.
    """
    if not encoder_name or ":" in str(encoder_name):
        return None
    try:
        import open_clip
        cfg = open_clip.get_model_config(str(encoder_name))
    except Exception:
        return None
    dim = (cfg or {}).get("embed_dim")
    return int(dim) if dim else None


def _selector_encoder(selector) -> Optional[dict]:
    """``selector._cache_config()``: the encoder whose embeddings the
    selector reads, or ``None`` for selectors that read none (``uniform``
    only uses the pool positions)."""
    fn = getattr(selector, "_cache_config", None)
    if not callable(fn):
        return None
    try:
        return fn()
    except Exception:
        return None


def _pool_was_capped(cache: dict, meta: dict) -> bool:
    """True when the cached pool is a subsample of the video's
    ``candidate_fps`` grid, i.e. it was built with a ``candidate_frames``
    cap below the grid size. ``False`` when the file records no fps."""
    native, cfps = meta.get("native_fps"), meta.get("candidate_fps")
    if not native or not cfps:
        return False
    step = max(1, int(round(float(native) / float(cfps))))
    n = int(cache["n_candidates"])
    if meta.get("total_frames") is not None:
        return n < len(range(0, int(meta["total_frames"]), step))
    # No frame count: an uncapped pool is exactly the 0, step, 2*step, ... grid.
    idx = np.asarray(cache["candidate_indices"])
    return not np.array_equal(idx, np.arange(len(idx)) * step)


def cache_mismatches(cache: dict, selector,
                     query_dim: Optional[int] = None) -> list[str]:
    """Reasons the loaded ``cache`` cannot serve ``selector`` (empty if none).

    Pool checks apply to every selector, since all of them pick from the
    cached candidate pool. Encoder checks apply to selectors that declare an
    encoder (``_cache_config()``), whose embedding fast path reads the
    cached values. ``query_dim`` is the width of the precomputed query
    embeddings the run will pass, if any; it is checked for every selector
    handed them, including ones that declare no encoder.
    """
    meta = cache.get("meta") or {}
    problems = []

    n = int(cache["n_candidates"])
    cap = getattr(selector, "candidate_frames", None) or None
    sel_fps = float(getattr(selector, "candidate_fps", None) or 1.0)
    cache_fps = meta.get("candidate_fps")
    if cap is not None and n > cap:
        problems.append(
            f"it holds {n} candidates per video, more than the selector's "
            f"candidate_frames={cap}"
        )
    if cache_fps is not None and abs(float(cache_fps) - sel_fps) > 1e-6:
        problems.append(
            f"its pool was sampled at candidate_fps={float(cache_fps):g}, the "
            f"selector uses candidate_fps={sel_fps:g}"
        )
    # An empty pool is a broken file, not a pool-size mismatch: leave it to
    # selection to report for that one video.
    elif n > 0 and (cap is None or n < cap) and _pool_was_capped(cache, meta):
        problems.append(
            f"its pool was capped at {n} candidates, the selector asks for "
            + (f"candidate_frames={cap}" if cap else "an uncapped pool")
        )

    cfg = _selector_encoder(selector)
    if not cfg and query_dim is None:
        return problems
    dim = int(np.asarray(cache["embeddings"]).shape[-1])
    want = None
    if cfg:
        enc, pre = cfg.get("encoder"), cfg.get("pretrained")
        rec_enc, rec_pre = meta.get("encoder"), meta.get("pretrained")
        if rec_enc not in _UNRECORDED and rec_enc != enc:
            problems.append(f"it was encoded with {rec_enc}, the selector uses {enc}")
        elif rec_pre not in _UNRECORDED and pre and rec_pre != pre:
            problems.append(
                f"it was encoded with {enc} '{rec_pre}' weights, the selector "
                f"uses '{pre}'"
            )
        want = encoder_output_dim(enc)
        if want is not None and dim != want:
            problems.append(f"its embeddings are {dim}-d, {enc} produces {want}-d")
    if (query_dim is not None and dim != int(query_dim)
            and (want is None or dim == want)):
        problems.append(
            f"its embeddings are {dim}-d, the query embeddings are {query_dim}-d"
        )
    return problems


# (cache directory, encoder, pretrained) already warned about.
_WARNED_UNRECORDED: set = set()


def check_cache_matches_selector(cache: dict, selector, cache_path: str, *,
                                 query_dim: Optional[int] = None) -> None:
    """Raise :class:`CacheMismatchError` if ``cache`` was built for another
    encoder or candidate pool than ``selector`` uses.

    A cache that records no encoder cannot be checked on that axis; the
    first such file per (directory, encoder) logs a warning instead.
    """
    problems = cache_mismatches(cache, selector, query_dim=query_dim)
    if problems:
        name = getattr(selector, "method_name", type(selector).__name__)
        raise CacheMismatchError(
            f"Embedding cache {cache_path} does not match selector {name!r}: "
            + "; ".join(problems)
            + ". The evaluator cache holds one <video>.npz per video, so its "
            "directory fixes the encoder and candidate pool: use one cache_dir "
            "per encoder x pool size (e.g. a fresh directory for this "
            "selector's encoder and candidate_frames)."
        )
    cfg = _selector_encoder(selector)
    if cfg and (cache.get("meta") or {}).get("encoder") in _UNRECORDED:
        key = (os.path.dirname(cache_path), cfg.get("encoder"), cfg.get("pretrained"))
        if key not in _WARNED_UNRECORDED:
            _WARNED_UNRECORDED.add(key)
            logger.warning(
                "Embedding cache %s records no encoder, so only its candidate "
                "pool and embedding width can be checked against %s (%s). Make "
                "sure the directory holds embeddings from that encoder: one "
                "cache_dir per encoder x pool size.",
                key[0], key[1], key[2],
            )
