# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Video I/O utilities.

Accepts a wide range of input types so users can feed whatever they
already have (path, URL, numpy, torch, list of PIL, `decord` reader)
without pre-converting. All inputs are normalized to the tuple
``(frames, original_indices, info)`` that the rest of the library
consumes.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union
from urllib.parse import urlparse

import numpy as np

logger = logging.getLogger(__name__)

# Default candidate FPS — pre-sample at 1 frame per second.
# Adjacent frames at 24-30 FPS are nearly identical, so encoding all of
# them through CLIP is wasteful.  1 FPS gives ~1 candidate per second,
# which is a good balance between coverage and speed.
DEFAULT_CANDIDATE_FPS = 1.0

# Download safety limits for remote http(s)/ftp URLs. A benchmark video
# is at most a few GB; the ceiling guards against an unbounded / hostile
# response body exhausting disk. Timeouts guard against a stalled server.
_DOWNLOAD_TIMEOUT_SEC = 30.0
_DOWNLOAD_MAX_BYTES = 8 * (1 << 30)  # 8 GiB
_DOWNLOAD_CHUNK = 1 << 20  # 1 MiB


# Public type alias — what `BaseSelector.select()` accepts as the video.
# At runtime we keep it broad (``object`` covers the duck-typed
# torch.Tensor / decord.VideoReader without forcing those imports). Under
# a type checker we narrow it to the real accepted union so ``select()``'s
# signature documents what it takes instead of collapsing to ``object``.
if TYPE_CHECKING:
    from decord import VideoReader
    from PIL.Image import Image as PILImage
    from torch import Tensor as TorchTensor

    VideoInput = Union[
        str,
        Path,
        np.ndarray,
        TorchTensor,
        List[PILImage],
        VideoReader,
    ]
else:
    VideoInput = Union[
        str,
        Path,
        np.ndarray,
        # torch.Tensor, list[PIL.Image.Image], decord.VideoReader
        # — runtime duck-typed, kept out of the type alias to avoid imports.
        object,
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FSSPEC_SCHEMES = {"s3", "gs", "gcs", "abfs", "az", "adl"}
_URL_SCHEMES = {"http", "https", "ftp"} | _FSSPEC_SCHEMES


def _is_url(path: str) -> bool:
    parsed = urlparse(path)
    return parsed.scheme in _URL_SCHEMES


def _assert_public_host(host: Optional[str]) -> None:
    """Reject hosts that resolve to loopback / link-local / private ranges.

    Guards against SSRF for http(s)/ftp downloads — a caller-supplied URL
    could otherwise reach the cloud instance-metadata endpoint
    (``169.254.169.254``), ``localhost``, or an internal RFC-1918 address.
    Every resolved address for ``host`` must be global.
    """
    if not host:
        raise ValueError("refusing to download from a URL with no host")

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise ValueError(f"cannot resolve host {host!r}: {e}") from e

    for info in infos:
        addr = info[4][0]
        # Strip an IPv6 scope/zone id (e.g. "fe80::1%eth0") before parsing.
        addr = addr.split("%", 1)[0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            raise ValueError(f"host {host!r} resolved to unparseable address {addr!r}")
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise ValueError(
                f"refusing to download from {host!r}: resolves to non-public "
                f"address {addr} (SSRF guard)"
            )


def _download_video(url: str) -> str:
    """Download ``url`` to a temporary file and return the local path.

    Uses ``fsspec`` for cloud schemes (s3, gs, ...) when available and
    falls back to ``urllib`` for http(s)/ftp.

    For http(s)/ftp the download is hardened against SSRF and unbounded
    responses: the scheme is checked against an allowlist, the host must
    resolve to a public address, connect/read use a timeout, and the body
    is streamed with a hard byte ceiling.
    """
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()

    if parsed.scheme in _FSSPEC_SCHEMES:
        try:
            import fsspec
        except ImportError as e:
            raise ImportError(
                f"Reading {parsed.scheme}:// URLs requires fsspec. "
                "Install with: pip install 'frameselect[cloud]'"
            ) from e
        with fsspec.open(url, "rb") as f_in, open(tmp.name, "wb") as f_out:
            # chunked copy to avoid loading whole files into memory
            while True:
                chunk = f_in.read(_DOWNLOAD_CHUNK)
                if not chunk:
                    break
                f_out.write(chunk)
        return tmp.name

    # http(s) / ftp — validated, bounded, timed download.
    if parsed.scheme not in {"http", "https", "ftp"}:
        # Should be unreachable (callers gate on _is_url), but fail closed.
        raise ValueError(f"unsupported URL scheme for download: {parsed.scheme!r}")

    _assert_public_host(parsed.hostname)

    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_SEC) as resp:
            total = 0
            with open(tmp.name, "wb") as f_out:
                while True:
                    chunk = resp.read(_DOWNLOAD_CHUNK)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > _DOWNLOAD_MAX_BYTES:
                        raise ValueError(
                            f"download from {parsed.hostname!r} exceeded the "
                            f"{_DOWNLOAD_MAX_BYTES} byte ceiling"
                        )
                    f_out.write(chunk)
    except Exception:
        # Don't leak the temp file if the download failed partway.
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise

    return tmp.name


def _capped_positions(
    n_candidates: int, max_candidates: Optional[int]
) -> np.ndarray:
    """Positions kept when a pool of ``n_candidates`` is capped at ``max_candidates``.

    The one cap rule shared by ``Selector.select()``, the selector cache and
    every evaluator path (cached, uncached, cache precompute): a pool larger
    than the cap keeps ``max_candidates`` positions spread uniformly over
    the whole pool, ``np.linspace(0, n - 1, cap).round()``, so a long video
    is covered end to end rather than cut to its first minutes. ``None`` or
    ``0`` disables the cap.
    """
    if max_candidates is not None and max_candidates < 0:
        raise ValueError(f"max_candidates must be >= 0, got {max_candidates}")
    if max_candidates and n_candidates > max_candidates:
        return np.linspace(0, n_candidates - 1, max_candidates).round().astype(int)
    return np.arange(n_candidates)


def _frames_from_array(
    frames: np.ndarray,
    candidate_fps: Optional[float],
    max_candidates: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Normalize an ``(T, H, W, 3)`` uint8 array to the canonical tuple.

    In-memory inputs don't carry native FPS, so ``candidate_fps`` only
    affects sub-sampling: ``None`` passes frames through as-is, a
    positive value sub-samples uniformly to approximate that rate
    assuming the input was already at "native" FPS. For in-memory
    frames, setting ``candidate_fps`` without a way to know the
    native rate is rarely what users want, so we default to keeping
    them as-is. ``max_candidates`` then caps the pool (see
    :func:`_capped_positions`).
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(
            f"Expected frames shape (T, H, W, 3), got {frames.shape}"
        )
    if frames.dtype != np.uint8:
        # Best-effort cast: float 0-1 or 0-255 both occur in the wild.
        if np.issubdtype(frames.dtype, np.floating):
            if frames.max() <= 1.0 + 1e-6:
                frames = (frames * 255.0).clip(0, 255).astype(np.uint8)
            else:
                frames = frames.clip(0, 255).astype(np.uint8)
        else:
            frames = frames.astype(np.uint8)

    total = len(frames)
    # ``candidate_fps`` is a no-op for in-memory frames: without a native
    # frame rate (arrays/tensors/PIL lists don't carry one) we can't map a
    # target FPS to a stride, so every frame is kept regardless. Log once so
    # a caller passing candidate_fps here isn't misled into thinking it
    # sub-sampled. (Use a real video path/decord reader to sub-sample.)
    if candidate_fps is not None and candidate_fps > 0 and total > 0:
        logger.debug(
            "candidate_fps=%s ignored for in-memory input (no native fps to "
            "sub-sample against); keeping all %d frames",
            candidate_fps,
            total,
        )
    indices = _capped_positions(total, max_candidates)
    if len(indices) < total:
        frames = frames[indices]

    info = {
        "source": "array",
        "total_frames": total,
        "native_fps": None,
        "duration_sec": None,
        "candidate_fps_used": candidate_fps,
        "n_candidates": len(indices),
        "n_candidates_uncapped": total,
    }
    return frames, indices, info


def _frames_from_torch(
    tensor, candidate_fps, max_candidates=None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Accept ``(T, C, H, W)`` or ``(T, H, W, C)`` torch tensor."""
    import torch

    assert isinstance(tensor, torch.Tensor)
    if tensor.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got shape {tuple(tensor.shape)}")

    # Detect channel-first vs channel-last
    # Channels are 1 or 3; spatial dims are typically much larger.
    if tensor.shape[1] in (1, 3) and tensor.shape[1] < min(tensor.shape[2], tensor.shape[3]):
        # (T, C, H, W) → (T, H, W, C)
        tensor = tensor.permute(0, 2, 3, 1).contiguous()
    elif tensor.shape[-1] in (1, 3):
        # already (T, H, W, C)
        pass
    else:
        raise ValueError(
            f"Cannot detect channel layout from tensor shape {tuple(tensor.shape)}. "
            "Expected (T, C, H, W) or (T, H, W, C) with C ∈ {1, 3}."
        )

    arr = tensor.detach().cpu().numpy()
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return _frames_from_array(arr, candidate_fps, max_candidates)


def _frames_from_pil_list(
    pil_list, candidate_fps, max_candidates=None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    from PIL import Image

    arrs = []
    for p in pil_list:
        if not isinstance(p, Image.Image):
            raise TypeError(
                f"All items in the list must be PIL.Image.Image, got {type(p).__name__}"
            )
        a = np.array(p.convert("RGB"))
        arrs.append(a)
    if not arrs:
        raise ValueError("Empty list of PIL images")
    # Ensure same shape
    h0, w0 = arrs[0].shape[:2]
    arrs = [a if a.shape[:2] == (h0, w0) else np.array(Image.fromarray(a).resize((w0, h0))) for a in arrs]
    frames = np.stack(arrs, axis=0)
    return _frames_from_array(frames, candidate_fps, max_candidates)


def _frames_from_decord(
    vr, candidate_fps, max_candidates=None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Accept an already-opened ``decord.VideoReader``.

    The pool is capped before decoding, so only the kept frames are read.
    """
    total = len(vr)
    native_fps = float(vr.get_avg_fps())
    duration = total / native_fps if native_fps > 0 else 0.0

    if candidate_fps is None:
        fps = DEFAULT_CANDIDATE_FPS
    elif candidate_fps <= 0:
        fps = native_fps
    else:
        fps = candidate_fps

    if fps < native_fps:
        step = max(1, int(round(native_fps / fps)))
        indices = np.arange(0, total, step)
    else:
        indices = np.arange(total)
    n_uncapped = len(indices)
    indices = indices[_capped_positions(n_uncapped, max_candidates)]

    frames = vr.get_batch(indices.tolist()).asnumpy()

    info = {
        "source": "decord.VideoReader",
        "total_frames": total,
        "native_fps": round(native_fps, 2),
        "duration_sec": round(duration, 2),
        "candidate_fps_used": round(fps, 2),
        "n_candidates": len(indices),
        "n_candidates_uncapped": n_uncapped,
    }
    return frames, indices, info


def _frames_from_path(
    path: str, candidate_fps, max_candidates=None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Open a local video path with decord and pre-sample."""
    from decord import VideoReader, cpu

    vr = VideoReader(path, ctx=cpu(0))
    frames, indices, info = _frames_from_decord(vr, candidate_fps, max_candidates)
    info["source"] = path
    return frames, indices, info


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_video_frames(
    video: VideoInput,
    *,
    candidate_fps: Optional[float] = None,
    max_candidates: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load candidate frames from heterogeneous input types.

    Accepts:

    * ``str`` or ``pathlib.Path`` — local file path, or a URL
      (``http://``, ``https://``, ``ftp://``, ``s3://``, ``gs://``, ...).
      Remote URLs are downloaded to a temp file.
    * ``numpy.ndarray`` of shape ``(T, H, W, 3)``. uint8 pass-through;
      float arrays are cast (assumed in ``[0, 1]`` or ``[0, 255]``).
    * ``torch.Tensor`` of shape ``(T, C, H, W)`` or ``(T, H, W, C)``.
    * ``list[PIL.Image.Image]``.
    * An already-opened ``decord.VideoReader``.

    Parameters
    ----------
    candidate_fps : float, optional
        Sampling rate of the candidate grid (video inputs only).
    max_candidates : int, optional
        Cap on the pool size. When the ``candidate_fps`` grid holds more
        candidates, ``max_candidates`` of them are kept, spread uniformly
        over the whole video (the evaluator's rule); for video inputs only
        the kept frames are decoded. ``None`` or ``0`` keeps every
        candidate.

    Returns
    -------
    frames : np.ndarray, shape (N, H, W, 3), uint8
        Pre-sampled candidate frames.
    original_indices : np.ndarray of int, shape (N,)
        The original frame indices in the source video. For in-memory
        inputs these are positions in the input sequence
        (``np.arange(T)`` when the cap does not bind).
    info : dict
        Metadata: ``source``, ``total_frames``, ``native_fps``,
        ``duration_sec``, ``candidate_fps_used``, ``n_candidates`` (pool
        size after the cap) and ``n_candidates_uncapped`` (before it).
    """
    # Path / URL --------------------------------------------------------
    if isinstance(video, (str, Path)):
        s = str(video)
        if _is_url(s):
            local = _download_video(s)
            try:
                return _frames_from_path(local, candidate_fps, max_candidates)
            finally:
                try:
                    os.unlink(local)
                except OSError:
                    pass
        else:
            return _frames_from_path(s, candidate_fps, max_candidates)

    # Numpy -------------------------------------------------------------
    if isinstance(video, np.ndarray):
        return _frames_from_array(video, candidate_fps, max_candidates)

    # Torch -------------------------------------------------------------
    try:
        import torch
        if isinstance(video, torch.Tensor):
            return _frames_from_torch(video, candidate_fps, max_candidates)
    except ImportError:
        pass

    # decord.VideoReader ------------------------------------------------
    # Duck-type check: decord doesn't expose a stable isinstance target
    # across versions, and we don't want to import it just to check.
    if hasattr(video, "get_batch") and hasattr(video, "get_avg_fps"):
        return _frames_from_decord(video, candidate_fps, max_candidates)

    # List of PIL -------------------------------------------------------
    if isinstance(video, (list, tuple)) and len(video) > 0:
        # Peek at first element
        head = video[0]
        if head.__class__.__module__.startswith("PIL"):
            return _frames_from_pil_list(video, candidate_fps, max_candidates)
        # list of numpy arrays → stack
        if isinstance(head, np.ndarray):
            arr = np.stack(list(video), axis=0)
            return _frames_from_array(arr, candidate_fps, max_candidates)

    raise TypeError(
        f"Unsupported video input type: {type(video).__name__}. "
        "Expected str/Path (file or URL), np.ndarray, torch.Tensor, "
        "list[PIL.Image], or decord.VideoReader."
    )
