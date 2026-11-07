# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Generic OpenAI-compatible VLM backend.

Lets FrameSelect talk to any server exposing the OpenAI
``/v1/chat/completions`` schema — vLLM, Ollama, LM Studio, TGI,
Fireworks, Together, DeepInfra, OpenAI itself. Useful for users who
want to run a local open-weights VLM (Qwen2.5-VL, LLaVA, etc.)
instead of Claude via Bedrock / Anthropic.

Activation
----------
Set two env vars:

- ``OPENAI_BASE_URL`` — e.g. ``http://localhost:8000/v1`` for a
  local vLLM server, or ``https://api.openai.com/v1`` for OpenAI.
- ``OPENAI_MODEL`` — model name the server expects. For vLLM this
  is the HuggingFace repo id, e.g. ``Qwen/Qwen2.5-VL-7B-Instruct``.

Optionally set ``OPENAI_API_KEY`` (most self-hosted servers accept
any non-empty string).

The evaluator auto-detects this backend first — if
``OPENAI_BASE_URL`` is set in the environment,
an evaluator such as :class:`ActivityNetEvaluator` will route through here.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
import time
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _probe_local_vllm(
    host: str = "localhost",
    port: int = 8000,
    timeout: float = 1.5,
) -> Optional[tuple[str, str]]:
    """Probe ``http://<host>:<port>/v1/models`` for a running vLLM server.

    Returns ``(base_url, first_model_id)`` if a server answers, else ``None``.
    Fast: uses a 1.5s socket timeout so it doesn't block startup.
    """
    import urllib.request
    import urllib.error
    url = f"http://{host}:{port}/v1/models"
    try:
        req = urllib.request.Request(url, headers={"Authorization": "Bearer EMPTY"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            import json as _json
            data = _json.loads(resp.read())
            models = data.get("data", [])
            if models:
                model_id = models[0].get("id", "")
                return f"http://{host}:{port}/v1", model_id
    except Exception:
        pass
    return None


def _default_base_url() -> Optional[str]:
    """Resolve a base URL from env var or local-vLLM auto-probe."""
    explicit = os.environ.get("OPENAI_BASE_URL")
    if explicit:
        return explicit
    # Auto-detect: probe the default vLLM port.
    probe = _probe_local_vllm()
    if probe is not None:
        return probe[0]
    return None


def _default_model_id() -> str:
    """Resolve a default model ID from env var or local-vLLM auto-probe."""
    explicit = os.environ.get("OPENAI_MODEL")
    if explicit:
        return explicit
    # Auto-detect: use the first model the local server advertises.
    probe = _probe_local_vllm()
    if probe is not None:
        return probe[1]
    return "Qwen/Qwen2.5-VL-7B-Instruct"


def _default_api_key() -> str:
    # vLLM / Ollama etc. accept any string; OpenAI needs a real key.
    return os.environ.get("OPENAI_API_KEY", "EMPTY")


# Backwards-compat alias kept for older import blocks.
HAIKU_MODEL_ID = _default_model_id()


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def _encode_frame(frame: np.ndarray, max_size: Optional[int] = None) -> str:
    """Encode a frame as a base64 JPEG data URI fragment (sans prefix).

    ``max_size`` controls the longest edge before encoding (in pixels).
    If ``None``, no resize is applied — the frame is sent at whatever
    resolution decord returned (typically 720p--1080p).
    The VLM's own processor will then resize per its native policy
    (e.g. 384x384 for LLaVA-OneVision via SigLIP, smart_resize for
    Qwen2.5-VL).

    Setting an explicit ``max_size`` pre-resizes before encoding. This
    is useful for (a) reducing bandwidth and (b) matching a specific
    paper's resolution choice when reproducing its numbers. Common
    values:

    - ``336`` — CLIP-ViT-L/14@336 native resolution
    - ``384`` — SigLIP-SO400M-384 native (LLaVA-OneVision)
    - ``512`` — the library default (``image_size=512``)
    - ``672`` — matches Qwen2.5-VL smart_resize output for 720p video
    """
    img = Image.fromarray(frame)
    if max_size is not None:
        ratio = min(max_size / img.width, max_size / img.height)
        if ratio < 1:
            img = img.resize((int(img.width * ratio), int(img.height * ratio)),
                             Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_mcq_message(
    frames: np.ndarray,
    question: str,
    options: list[str],
    image_size: Optional[int] = None,
) -> list[dict]:
    """Build OpenAI-compatible messages (vision-capable)."""
    content = []
    for f in frames:
        b64 = _encode_frame(f, max_size=image_size)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    from ._mcq_parse import mcq_answer_instruction

    option_str = "\n".join(options)
    content.append({
        "type": "text",
        "text": (
            "These are frames sampled from a video. "
            "Answer the following multiple choice question.\n\n"
            f"Question: {question}\n"
            f"Options:\n{option_str}\n\n"
            f"{mcq_answer_instruction(len(options))}"
        ),
    })

    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_client = None


def _get_client():
    """Lazy-init the OpenAI client pointed at our server."""
    global _client
    if _client is not None:
        return _client

    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "openai package not installed. Install with: pip install openai"
        ) from e

    base_url = _default_base_url()
    if not base_url:
        raise RuntimeError(
            "No OpenAI-compatible VLM server found.\n"
            "\n"
            "Options:\n"
            "  1. Start a local vLLM server (auto-detected on port 8000):\n"
            "       vllm serve Qwen/Qwen2.5-VL-7B-Instruct --host 127.0.0.1 --port 8000\n"
            "\n"
            "  2. Point an evaluator or selector at an existing server:\n"
            "       ev  = ActivityNetEvaluator(\n"
            "           vlm='openai:Qwen/Qwen2.5-VL-7B-Instruct')\n"
            "       # or set via env var:\n"
            "       export OPENAI_BASE_URL=http://your-server:8000/v1\n"
            "       export OPENAI_MODEL=Qwen/Qwen2.5-VL-7B-Instruct\n"
            "\n"
            "  3. Use a different backend:\n"
            "       ev = ActivityNetEvaluator(vlm='hf:Qwen/Qwen2.5-VL-7B-Instruct')\n"
            "       ev = ActivityNetEvaluator(vlm='bedrock:<model>')\n"
            "       ev = ActivityNetEvaluator(vlm='anthropic:<model>')\n"
            "\n"
            "Run fs.check_backend() to diagnose the current environment."
        )

    _client = OpenAI(base_url=base_url, api_key=_default_api_key())
    return _client


def call_vlm(
    messages: list[dict],
    model_id: Optional[str] = None,
    max_retries: int = 3,
    n_options: Optional[int] = None,
) -> str:
    """Call an OpenAI-compatible server and return a single option letter.

    Uses ``max_tokens=1`` plus a vLLM-specific ``guided_choice``
    constraint so the server can stop after the very first decoded
    token. Non-vLLM servers silently ignore ``guided_choice`` via
    the ``extra_body`` channel and fall back to our regex parser —
    so the call still works against OpenAI, Ollama, TGI, etc.

    ``n_options`` sets the valid-letter range. For a 5-option question
    the ``guided_choice`` list becomes ``["A","B","C","D","E"]`` and the
    parser accepts up to ``E`` — without it a benchmark like ActivityNet
    (5 options) could never return ``E``. When ``None`` the range falls
    back to the historical ``A..D``.

    When a server returns more than one character (ignoring
    ``guided_choice``), the parser extracts an MCQ letter robustly:
    a bare letter (``"B"``), a labelled letter (``"Answer: B"``,
    ``"(B)"``), or the last word-boundaried valid letter — never the
    ``"A"`` embedded in the word ``"answer"``.

    On genuine failure (all retries exhausted or a non-recoverable
    error) returns :data:`FAILED_ANSWER` (the empty-string sentinel),
    NOT ``"A"``, so a failed call is distinguishable downstream.
    """
    from ._mcq_parse import FAILED_ANSWER, option_letters, parse_mcq_letter

    client = _get_client()
    model_id = model_id or _default_model_id()

    # Constrain the reply to one option letter: older vLLM reads
    # ``guided_choice``, newer vLLM ``structured_outputs``. Servers that
    # know neither ignore them; worst case we regex-parse one token.
    letters = option_letters(n_options)
    extra = {"guided_choice": letters, "structured_outputs": {"choice": letters}}

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=1,
                temperature=0,
                extra_body=extra,
            )
            text = response.choices[0].message.content or ""
            return parse_mcq_letter(text, n_options=n_options)
        except Exception as e:
            msg = str(e)
            # Some backends reject the extra_body key — retry without it.
            if "guided_choice" in msg or "structured_outputs" in msg or "extra_body" in msg:
                extra = {}
                continue
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.warning("OpenAI-compatible server call failed: %s",
                               type(e).__name__)
                return FAILED_ANSWER

    return FAILED_ANSWER


def _parse_mcq_letter(text: str) -> str:
    """Extract an MCQ answer letter (A/B/C/D) from a VLM response.

    Forwards to the shared parser in ``vlm._mcq_parse``. Kept as a
    module-local name so existing callers importing
    ``frameselect.vlm.openai._parse_mcq_letter`` still work.
    """
    from ._mcq_parse import parse_mcq_letter
    return parse_mcq_letter(text)


def call_vlm_row(row, model_id: Optional[str] = None) -> str:
    """Row-wise wrapper for pandas.

    Threads the per-question option count (``len(row["options"])``)
    through so ``guided_choice`` and the parser use the real letter
    range for 5--7-way benchmarks.
    """
    return call_vlm(
        row["messages"], model_id=model_id, n_options=_row_n_options(row)
    )


def _row_n_options(row) -> Optional[int]:
    """Best-effort option count from an evaluator row (``None`` if absent)."""
    try:
        opts = row["options"]
    except (KeyError, TypeError):
        return None
    try:
        return len(opts) if opts is not None else None
    except TypeError:
        return None


# Alias so older import blocks see the same symbol names regardless of backend.
call_haiku = call_vlm
call_haiku_row = call_vlm_row
