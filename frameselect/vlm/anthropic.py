# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Claude via the Anthropic API — sequential calls.

The last fallback when no other backend is configured. Requires:
- the ``anthropic`` package (a core dependency)
- the ``ANTHROPIC_API_KEY`` environment variable

Calls are sequential (one request at a time) to stay within rate limits.
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


HAIKU_MODEL_ID = "claude-haiku-4-5-20251001"


# Short catalog key → Anthropic-API model string. The Bedrock catalog
# uses the same short keys; this keeps the public ``model_id`` API
# consistent across backends.
_ANTHROPIC_MODEL_CATALOG: dict[str, str] = {
    "claude-haiku-4-5":  "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6": "claude-sonnet-4-6",
}


def _resolve_model_id(model_id: str) -> str:
    """Map a catalog short key to the full Anthropic model string.

    Full model strings (``claude-haiku-4-5-20251001``) are passed
    through unchanged.
    """
    return _ANTHROPIC_MODEL_CATALOG.get(model_id, model_id)


def _encode_frame(frame: np.ndarray, max_size: Optional[int] = 512) -> str:
    """Encode a frame as base64 JPEG, resized to fit within max_size.

    ``max_size=None`` (the evaluator's ``image_size=None``) sends the
    frame at native resolution and leaves resizing to the API.
    """
    img = Image.fromarray(frame)
    if max_size is not None:
        ratio = min(max_size / img.width, max_size / img.height)
        if ratio < 1:
            img = img.resize((int(img.width * ratio), int(img.height * ratio)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_mcq_message(
    frames: np.ndarray,
    question: str,
    options: list[str],
    image_size: Optional[int] = 512,
) -> list[dict]:
    """Build Claude messages with frames as images + MCQ prompt."""
    content = []
    for f in frames:
        b64 = _encode_frame(f, max_size=image_size)
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": b64,
            },
        })

    from ._mcq_parse import mcq_answer_instruction

    option_str = "\n".join(options)
    content.append({
        "type": "text",
        "text": (
            f"These are frames sampled from a video. "
            f"Answer the following multiple choice question.\n\n"
            f"Question: {question}\n"
            f"Options:\n{option_str}\n\n"
            f"{mcq_answer_instruction(len(options))}"
        ),
    })

    return [{"role": "user", "content": content}]


_client = None


def _get_client():
    """Lazy-init the Anthropic client."""
    global _client
    if _client is None:
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. "
                "Install with: pip install anthropic"
            )
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. "
                "Export your API key: export ANTHROPIC_API_KEY=sk-ant-..."
            )
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def call_haiku(
    messages: list[dict],
    model_id: str = HAIKU_MODEL_ID,
    max_retries: int = 3,
    n_options: Optional[int] = None,
) -> str:
    """Call Claude via the Anthropic API with retries.

    Robust answer-letter extraction via ``vlm._mcq_parse`` so responses
    like ``"The answer is B."`` return ``"B"``, not the ``"A"`` hidden
    inside ``"answer"``. ``n_options`` widens the valid-letter range so
    5--7-way benchmarks can return ``E``/``F``/``G``.

    On genuine failure (all retries exhausted) returns
    :data:`FAILED_ANSWER` (empty-string sentinel), NOT ``"A"``, so a
    failed call is distinguishable downstream.
    """
    from ._mcq_parse import FAILED_ANSWER, parse_mcq_letter
    client = _get_client()

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=_resolve_model_id(model_id),
                messages=messages,
                max_tokens=32,
                temperature=0,
            )
            text = response.content[0].text
            return parse_mcq_letter(text, n_options=n_options)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.warning("Anthropic API call failed: %s",
                               type(e).__name__)
                return FAILED_ANSWER

    return FAILED_ANSWER


def call_haiku_row(row, model_id=HAIKU_MODEL_ID):
    """Row-wise wrapper for compatibility with pandas apply.

    Threads the per-question option count through so the parser uses the
    real letter range for 5--7-way benchmarks.
    """
    return call_haiku(
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
