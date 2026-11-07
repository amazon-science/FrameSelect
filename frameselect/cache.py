# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""First-class embedding cache.

Re-encoding candidate frames through CLIP on every ``select()`` call is
wasteful: the work depends only on the video, the encoder and the
candidate pool, not on the selection method, frame budget, or query.
This module makes that cache a public API so end users get the same
speedup our benchmark runs enjoy.

Usage::

    import frameselect as fs

    sel = fs.create("efs", cache_dir="~/.cache/frameselect")
    # first call: encode + cache
    r1 = sel.select("video.mp4", n_frames=16, query="...")
    # second call on same video: skips CLIP forward
    r2 = sel.select("video.mp4", n_frames=16, query="...")

Or point at the dir via an env var::

    export FRAMESELECT_CACHE_DIR=~/.cache/frameselect

Cache layout on disk::

    <cache_dir>/<key>.npz              embeddings + candidate_indices + meta
    <cache_dir>/<key>_frames.bin       JPEG-encoded frame sidecar with offset table

The key hashes ``(video content, encoder_name, pretrained_tag,
candidate_fps)``. Neither ``n_frames`` nor the ``candidate_frames`` cap
(``max_candidates``) is part of the key: embeddings are independent of
the selection budget, and one entry is shared across caps. Each entry
records the pool size before the cap (``meta["n_candidates_uncapped"]``),
and ``select()`` reuses an entry only when it contains every candidate
of the pool an uncached call would build under the selector's cap (an
entry holding the whole ``candidate_fps`` grid serves any cap);
otherwise the entry is re-encoded and overwritten.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache dir resolution (XDG)
# ---------------------------------------------------------------------------

def default_cache_dir() -> Path:
    """Return the default on-disk cache directory.

    Precedence:
    1. ``$FRAMESELECT_CACHE_DIR`` env var if set
    2. ``$XDG_CACHE_HOME/frameselect`` if ``XDG_CACHE_HOME`` is set
    3. ``~/.cache/frameselect`` otherwise (XDG default on Linux/macOS)
    """
    env = os.environ.get("FRAMESELECT_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser() / "frameselect"
    return Path.home() / ".cache" / "frameselect"


# ---------------------------------------------------------------------------
# Key derivation
# ---------------------------------------------------------------------------

def _hash_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Content hash of a file. For speed on huge videos we hash only
    the first ``chunk_size`` bytes plus the file size + mtime.

    This isn't a cryptographic identity — it's a fingerprint. If a
    user replaces a video with another of the exact same size/mtime/head
    they get a false hit; in practice that's astronomically unlikely.
    """
    h = hashlib.sha256()
    stat = path.stat()
    h.update(struct.pack("<qq", stat.st_size, int(stat.st_mtime)))
    with open(path, "rb") as f:
        h.update(f.read(chunk_size))
    return h.hexdigest()[:24]


def make_key(
    video: "str | Path | object",
    *,
    encoder_name: str,
    encoder_pretrained: str,
    candidate_fps: float,
    max_candidates: Optional[int] = None,
) -> Optional[str]:
    """Derive a cache key for one video + encoder config.

    The key fingerprints ``(video_content, encoder_name,
    encoder_pretrained, candidate_fps)``. ``max_candidates`` is
    accepted for backwards compatibility but deliberately **not**
    part of the fingerprint: embeddings depend only on the video +
    encoder + sampling rate, not on how many frames a caller later
    chose to select. This lets ``n_frames=8`` and ``n_frames=16``
    runs reuse the same on-disk cache entry. (Callers that cap the
    pool check an entry's stored pool before reusing it; see the
    module docstring.)

    Returns ``None`` when the input can't be cached (in-memory arrays,
    streams, decord readers — we have nothing stable to hash).
    """
    del max_candidates  # accepted for back-compat, intentionally unused.

    if isinstance(video, (str, Path)):
        p = Path(video).expanduser()
        if not p.exists():
            # URL or non-existent — skip caching; caller falls back to
            # the no-cache path.
            return None
        content = _hash_file(p)
    else:
        # In-memory inputs don't have a stable identity. Caller should
        # not cache these. Returning None makes the cache a no-op.
        return None

    fingerprint = "|".join([
        content,
        encoder_name,
        encoder_pretrained,
        f"{candidate_fps:.4f}",
    ])
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]
    return f"{content}__{encoder_name.replace('/', '_')}__{digest}"


# ---------------------------------------------------------------------------
# On-disk entry
# ---------------------------------------------------------------------------

@dataclass
class CachedEntry:
    """In-memory view of a cache entry."""
    candidate_indices: np.ndarray
    embeddings: np.ndarray
    n_candidates: int
    frames_path: Optional[Path]   # sidecar JPEG file, may be None
    meta: dict

    def __len__(self) -> int:
        return self.n_candidates


def _save_frames_bin(path: Path, frames: np.ndarray, *, quality: int = 85) -> None:
    """Encode ``frames`` as a JPEG bag with an offset table at the head.

    Format (little-endian):
        [u32 n]
        [u64 off_0 ... off_{n-1}]  — absolute file offsets of each frame record
        for each frame i:
            [u32 length][bytes jpg]
    """
    from PIL import Image

    n = len(frames)
    # First: encode every frame into bytes, remembering sizes.
    blobs = []
    for f in frames:
        buf = io.BytesIO()
        Image.fromarray(f).save(buf, format="JPEG", quality=quality)
        blobs.append(buf.getvalue())

    # Compute offsets. Header = 4 + n*8 bytes. Each record adds 4 + len(blob).
    header_size = 4 + n * 8
    offsets = []
    off = header_size
    for b in blobs:
        offsets.append(off)
        off += 4 + len(b)

    with open(path, "wb") as f:
        f.write(struct.pack("<I", n))
        for o in offsets:
            f.write(struct.pack("<Q", o))
        for b in blobs:
            f.write(struct.pack("<I", len(b)))
            f.write(b)


def _load_frames_bin(path: Path) -> np.ndarray:
    from PIL import Image
    with open(path, "rb") as f:
        n = struct.unpack("<I", f.read(4))[0]
        offsets = list(struct.unpack(f"<{n}Q", f.read(8 * n)))
        frames = []
        for off in offsets:
            f.seek(off)
            length = struct.unpack("<I", f.read(4))[0]
            frames.append(np.array(Image.open(io.BytesIO(f.read(length)))))
    # Match the (N, H, W, 3) uint8 contract even when empty, so callers
    # can index/stack without special-casing a 1-D float64 array.
    return np.stack(frames) if frames else np.empty((0, 1, 1, 3), dtype=np.uint8)


class EmbeddingCache:
    """Disk cache for ``(candidate_indices, embeddings, frames)`` tuples.

    Thread-safe for single-writer / many-reader. Concurrent writers
    into the same key will race on rename but produce a valid file
    (last writer wins; all writers produce equivalent content for
    deterministic encoders).
    """

    def __init__(self, root: "str | Path | None" = None, *, save_frames: bool = True):
        if root is None:
            root = default_cache_dir()
        self.root = Path(root).expanduser()
        self.save_frames = save_frames
        self.root.mkdir(parents=True, exist_ok=True)

    # -- internal paths ----------------------------------------------

    def _npz_path(self, key: str) -> Path:
        return self.root / f"{key}.npz"

    def _bin_path(self, key: str) -> Path:
        return self.root / f"{key}_frames.bin"

    # -- public API --------------------------------------------------

    def has(self, key: str) -> bool:
        return self._npz_path(key).exists()

    def get(self, key: str) -> Optional[CachedEntry]:
        """Load the entry for ``key`` if it exists, else ``None``."""
        p = self._npz_path(key)
        if not p.exists():
            return None
        try:
            data = np.load(str(p), allow_pickle=True)
        except Exception:
            # corrupted file → treat as miss and let the caller re-encode.
            # Log first so a real bug (bad numpy, permission error) isn't
            # silently masked as a plain cache miss.
            logger.warning(
                "cache: failed to load npz %s; treating as a cache miss",
                p,
                exc_info=True,
            )
            return None
        meta = {}
        if "meta" in data.files:
            try:
                meta = dict(data["meta"].item())
            except Exception:
                meta = {}
        bin_p = self._bin_path(key)
        return CachedEntry(
            candidate_indices=np.asarray(data["candidate_indices"]),
            embeddings=np.asarray(data["embeddings"]),
            n_candidates=int(data["n_candidates"]),
            frames_path=bin_p if bin_p.exists() else None,
            meta=meta,
        )

    def put(
        self,
        key: str,
        *,
        candidate_indices: np.ndarray,
        embeddings: np.ndarray,
        frames: Optional[np.ndarray] = None,
        meta: Optional[dict] = None,
    ) -> None:
        """Write a cache entry atomically (via tmp-file + rename)."""
        meta = meta or {}
        # np.savez auto-appends .npz to the given path, so we write to
        # a .npz-suffixed tempfile and rename into place.
        final = self._npz_path(key)
        tmp_base = str(final) + ".tmp"   # will become tmp_base + ".npz"
        np.savez(
            tmp_base,
            candidate_indices=candidate_indices.astype(np.int64),
            embeddings=embeddings.astype(np.float32),
            n_candidates=np.array(len(candidate_indices), dtype=np.int64),
            meta=np.array(meta, dtype=object),
        )
        os.replace(tmp_base + ".npz", final)

        if self.save_frames and frames is not None:
            bin_final = self._bin_path(key)
            bin_tmp = str(bin_final) + ".tmp"
            _save_frames_bin(Path(bin_tmp), frames)
            os.replace(bin_tmp, bin_final)

    def load_frames(self, key: str) -> Optional[np.ndarray]:
        """Load all candidate frames for ``key``. Returns ``None`` if the
        sidecar is missing.
        """
        p = self._bin_path(key)
        if not p.exists():
            return None
        return _load_frames_bin(p)

    # -- maintenance -------------------------------------------------

    def stats(self) -> dict:
        """Summarize the cache: entry count + total size on disk."""
        npz = list(self.root.glob("*.npz"))
        bins = list(self.root.glob("*_frames.bin"))
        total_bytes = sum(p.stat().st_size for p in npz + bins)
        return {
            "root": str(self.root),
            "entries": len(npz),
            "frames_sidecars": len(bins),
            "size_mb": round(total_bytes / (1 << 20), 2),
        }

    def clear(self) -> int:
        """Delete all entries in the cache directory.

        Returns the number of files removed. Only removes files produced
        by this module (``*.npz`` and ``*_frames.bin`` under ``root``);
        anything else is left alone.
        """
        removed = 0
        for p in self.root.glob("*.npz"):
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
        for p in self.root.glob("*_frames.bin"):
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
        return removed


# ---------------------------------------------------------------------------
# Convenience — module-level functions operate on the default cache
# ---------------------------------------------------------------------------

_default: Optional[EmbeddingCache] = None


def _shared() -> EmbeddingCache:
    global _default
    if _default is None:
        _default = EmbeddingCache(default_cache_dir())
    return _default


def stats() -> dict:
    """Return stats for the default cache (see :class:`EmbeddingCache.stats`)."""
    return _shared().stats()


def clear() -> int:
    """Delete all entries in the default cache."""
    return _shared().clear()


def path() -> Path:
    """Return the default cache directory path."""
    return default_cache_dir()
