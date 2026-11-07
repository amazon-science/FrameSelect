# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""frameselect — a unified library for video frame selection and evaluation.

The core is task-agnostic. One API, 6 frame selectors, any video
source. Picked frames go wherever you want them — a VLM, a
captioning model, a retrieval pipeline, a notebook display, anything.

Quick start::

    import numpy as np
    from frameselect import Selector, Evaluator

    # Query-agnostic selector
    sel = Selector("uniform", n_frames=8)
    r = sel.select("video.mp4")

    # Query-aware selector with a SigLIP2 encoder
    sel = Selector("efs",
                   n_frames=8,
                   encoder="ViT-SO400M-16-SigLIP2-384",
                   encoder_pretrained="webli")
    r = sel.select("video.mp4", query="what happens?")

    # Score a selector on a benchmark
    ev = Evaluator("activitynet", video_dir="/path/to/activitynet/videos",
                   vlm="openai:Qwen/Qwen2.5-VL-7B-Instruct")
    r  = ev.run(sel)

    # Any input type: path, URL, np.ndarray, torch.Tensor,
    # list[PIL.Image], decord.VideoReader
    frames = np.zeros((240, 224, 224, 3), dtype=np.uint8)
    r = Selector("efs", n_frames=8).select(frames, query="...")

    r.frames   # (N, H, W, 3) uint8 — pass to any downstream system
    r.indices  # absolute frame indices in the source video
    r.scores   # per-picked-frame relevance

Direct-class imports still work (``from frameselect.evaluations
import ActivityNetEvaluator``) for users who want IDE autocomplete on
per-evaluator kwargs; the string factory is an additional convenient
path, not a replacement.

The ``frameselect.evaluations`` sub-module ships a reference evaluator
for ActivityNet activity recognition — one application built on top of
the library, not its purpose. Most users won't need it.

See ``examples/quickstart.ipynb`` for a runnable tour of the selectors.
"""

__version__ = "0.1.0"

from .base import BaseSelector, SelectionResult
from .factory import Evaluator, Selector
from .registry import list_methods
from .video_io import VideoInput, load_video_frames
from . import cache  # noqa: F401
from . import evaluations  # noqa: F401
from . import viz  # noqa: F401
from . import vlm  # noqa: F401

# Import methods to trigger registration
from . import methods as _methods  # noqa: F401


def create(method: str, device: str = "cpu", **kwargs) -> BaseSelector:
    """Create a frame selector by method name.

    Parameters
    ----------
    method : str
        Name of the selection method. See :func:`list_methods`.
    device : str
        PyTorch device string (``"cpu"`` or ``"cuda"``).
    **kwargs
        Unified evaluation-axis controls (all optional, with sensible
        defaults):

        * ``n_frames`` — output frame budget.
        * ``candidate_frames`` — max candidate-pool size (candidates
          kept uniformly over the whole video).
        * ``candidate_fps`` — candidate-pool sampling rate.
        * ``encoder`` / ``encoder_pretrained`` — frame encoder
          identifier (CLIP/SigLIP selectors only).
        * ``vlm`` — ``"backend:model"`` spec for selectors that call a
          VLM during selection (none of the bundled selectors does).
        * ``seed`` — random seed for stochastic methods.

        Plus any method-specific hyperparameters (see that method's
        class docstring, e.g. ``max_events`` for EFS, ``preset`` for AGFS).

    Returns
    -------
    BaseSelector
        An initialized selector ready to call ``.select()``.

    Examples
    --------
    >>> sel = fs.create("efs", device="cuda", max_events=10, n_frames=8)
    >>> result = sel.select("video.mp4", query="What happens?")
    """
    from .registry import get_method

    cls = get_method(method)
    return cls(device=device, **kwargs)


def check_backend() -> None:
    """Probe the configured answering-VLM backend and print a status report.

    Checks the environment for a usable VLM backend in priority order:

    1. ``OPENAI_BASE_URL`` (+ ``OPENAI_MODEL``) — OpenAI-compatible server.
    2. Auto-detected local vLLM at ``localhost:8000``.
    3. ``HF_VLM_MODEL`` — in-process HuggingFace model.
    4. AWS Bedrock with boto3's default credentials.
    5. ``ANTHROPIC_API_KEY`` — Anthropic direct.

    This is a diagnostic. To actually pick a VLM, pass
    ``vlm="backend:model"`` on the selector or evaluator you care
    about — there is no global VLM singleton.
    """
    import os
    from .vlm.openai import _probe_local_vllm

    lines: list[str] = []
    lines.append("VLM backend diagnostic (no global singleton is consulted):")
    lines.append("")

    # openai env var
    env_url = os.environ.get("OPENAI_BASE_URL")
    if env_url:
        probe = _probe_local_vllm(*_parse_host_port(env_url))
        mark = "✓" if probe else "✗"
        lines.append(f"  {mark}  OPENAI_BASE_URL={env_url}")
        if probe:
            lines.append(f"       model: {probe[1]}")
        else:
            lines.append("       server not reachable")
    else:
        lines.append("  —   OPENAI_BASE_URL not set")

    # auto-probe localhost:8000
    probe = _probe_local_vllm()
    if probe:
        lines.append(f"  ✓  auto-detected vLLM at {probe[0]}")
        lines.append(f"       model: {probe[1]}")
    else:
        lines.append("  ✗  localhost:8000 — no vLLM server found")

    # HF
    hf_model = os.environ.get("HF_VLM_MODEL")
    if hf_model:
        lines.append(f"  ✓  HF_VLM_MODEL={hf_model}")
    else:
        lines.append("  —   HF_VLM_MODEL not set")

    # Bedrock
    try:
        import boto3
        from .vlm.bedrock import ACTIVE_REGIONS
        if boto3.Session().get_credentials() is not None:
            lines.append(
                f"  ✓  Bedrock — boto3 default credentials, region(s): {', '.join(ACTIVE_REGIONS)}"
            )
        else:
            lines.append("  —   Bedrock — no AWS credentials found")
    except Exception:
        lines.append("  —   Bedrock — not available")

    # Anthropic
    if os.environ.get("ANTHROPIC_API_KEY"):
        lines.append("  ✓  Anthropic — ANTHROPIC_API_KEY set")
    else:
        lines.append("  —   Anthropic — ANTHROPIC_API_KEY not set")

    lines.append("")
    if not probe and not env_url and not hf_model:
        lines.append("No local VLM server or HF model is configured. Start a local vLLM server:")
        lines.append("    vllm serve Qwen/Qwen2.5-VL-7B-Instruct --host 127.0.0.1 --port 8000")
        lines.append("")
        lines.append("Or configure a selector/evaluator explicitly:")
        lines.append("    ev  = ActivityNetEvaluator(vlm='hf:Qwen/Qwen2.5-VL-7B-Instruct', ...)")

    print("\n".join(lines))


def _parse_host_port(url: str) -> tuple[str, int]:
    """Extract (host, port) from a URL string for ``_probe_local_vllm``."""
    try:
        from urllib.parse import urlparse
        p = urlparse(url)
        host = p.hostname or "localhost"
        port = p.port or 8000
        return host, port
    except Exception:
        return "localhost", 8000


__all__ = [
    # Class-style factories (recommended)
    "Selector",
    "Evaluator",
    # Function-style factory
    "create",
    # Diagnostics
    "check_backend",
    "list_methods",
    # Abstract bases
    "BaseSelector",
    "SelectionResult",
    # Video IO
    "VideoInput",
    "load_video_frames",
    # Subpackages
    "cache",
    "evaluations",
    "viz",
    "vlm",
    "__version__",
]
