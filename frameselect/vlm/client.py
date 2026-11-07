# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Backend-agnostic VLM client for selectors that call a VLM during selection.

Such selectors pass their ``vlm_spec`` — set at construction via
``fs.create("<name>", vlm="bedrock:...")`` — to every call. That spec fully determines the backend + model for this call,
without touching any global state. An explicit but unknown backend
prefix raises ``ValueError`` (see :func:`_parse_vlm_spec`). When no spec
is passed we fall back to env-var detection (same priority as
:mod:`frameselect.evaluations.base`).

Backend resolution order when no ``vlm_spec`` is provided:

1. ``OPENAI_BASE_URL`` → any OpenAI-compatible server (vLLM, Ollama, OpenAI, etc.)
2. ``HF_VLM_MODEL``    → local HuggingFace model (loaded in-process)
3. ``boto3`` importable → AWS Bedrock
4. Anthropic API fallback

Exposes two callables:
- :func:`call_text` for text-only prompts.
- :func:`call_vision` for prompts with a single PIL image + text.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

# Botocore error codes that mean the cached client's credentials are no
# longer valid and it must be rebuilt.
_EXPIRED_ERROR_CODES = frozenset({
    "ExpiredToken",
    "ExpiredTokenException",
    "RequestExpired",
})


# -----------------------------------------------------------------------------
# Backend detection — env vars only. Selectors pass their own spec via
# ``vlm_spec=`` on every call when they want a non-default.
# -----------------------------------------------------------------------------

_KNOWN_BACKENDS = ("openai", "hf", "bedrock", "anthropic")

# A backend prefix is a bare word before the first ":" ("bedrok:...").
# Model ids that merely contain a colon are not prefixes: dotted or
# slashed ids never match, and a purely numeric tail is a version suffix
# ("claude-haiku-4-5-20251001-v1:0").
_PREFIX_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_-]*):(.*)$", re.DOTALL)


def _parse_vlm_spec(spec: str) -> tuple[str, str]:
    """Parse ``"backend:model"`` → ``(backend, model)``.

    Prefix-less strings are auto-detected: ids containing ``/`` are
    treated as HuggingFace repos, anything else as an Anthropic model ID
    — callers should pass an explicit prefix.

    Raises
    ------
    ValueError
        If the spec carries an explicit prefix that is not a known
        backend (e.g. a typo such as ``"bedrok:<model>"``),
        instead of silently routing it to another backend.
    """
    s = spec.strip()
    for backend in _KNOWN_BACKENDS:
        prefix = backend + ":"
        if s.lower().startswith(prefix):
            return backend, s[len(prefix):]
    m = _PREFIX_RE.match(s)
    if m and not m.group(2).isdigit():
        known = ", ".join(f"'{b}:'" for b in _KNOWN_BACKENDS)
        raise ValueError(
            f"Unknown VLM backend prefix {m.group(1)!r} in spec {spec!r}. "
            f"Known prefixes: {known}; omit the prefix to auto-detect."
        )
    # No prefix — best guess based on common patterns.
    if "/" in s:
        return "hf", s
    return "anthropic", s


def _resolve_selection_backend(vlm_spec: Optional[str] = None):
    """Resolve the active VLM backend + default model for this call.

    When ``vlm_spec`` is provided, parse it and use the requested
    backend. Otherwise fall back to env-var detection (the same
    priority order as ``evaluations/base.py`` uses for the answering
    VLM). Resolved fresh on every request so env-var tweaks take
    effect without a reimport.
    """
    if vlm_spec is not None:
        return _parse_vlm_spec(vlm_spec)
    # Env-var fallback
    if os.environ.get("OPENAI_BASE_URL"):
        return "openai", os.environ.get("OPENAI_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    if os.environ.get("HF_VLM_MODEL"):
        return "hf", os.environ["HF_VLM_MODEL"]
    try:
        from .bedrock import HAIKU_MODEL_ID as _hid
        return "bedrock", _hid
    except ImportError:
        return "anthropic", "claude-haiku-4-5-20251001"


# Legacy module-level constant. Kept for back-compat with code that
# did ``from .client import _BACKEND``. Reflects the
# default-backend (env-var) choice at import time; per-call overrides
# happen via the ``vlm_spec`` argument below.
_BACKEND, _DEFAULT_MODEL_ID = _resolve_selection_backend()

# Keep bedrock imports if needed
if _BACKEND == "bedrock":
    try:
        from .bedrock import (
            _get_all_endpoints,
            _get_client,
            _REGION_TO_PROFILE,
            _bedrock_clients,
        )
    except ImportError:
        _BACKEND = "anthropic"
        _DEFAULT_MODEL_ID = "claude-haiku-4-5-20251001"
        _REGION_TO_PROFILE = {}
else:
    _REGION_TO_PROFILE = {}


# Per-process region cycle: each call starts at the next configured
# region (see ``BEDROCK_REGIONS``).
import itertools
import threading
_ep_lock = threading.Lock()
_ep_cycle = None  # filled lazily when first Bedrock call is made


def _pil_to_b64(img: Image.Image, max_size: int = 1024) -> str:
    """Base64-encode a PIL image (JPEG, bounded by ``max_size``)."""
    ratio = min(max_size / img.width, max_size / img.height)
    if ratio < 1:
        img = img.resize((int(img.width * ratio), int(img.height * ratio)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# -----------------------------------------------------------------------------
# OpenAI-compatible path
# -----------------------------------------------------------------------------
def _call_openai(messages: list[dict], max_tokens: int, model_id: str) -> str:
    from .openai import _get_client
    client = _get_client()
    model_id = model_id or os.environ.get("OPENAI_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    import time as _time
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            if attempt < 2:
                _time.sleep(2 ** attempt)
            else:
                raise


# -----------------------------------------------------------------------------
# HuggingFace path
# -----------------------------------------------------------------------------
def _call_hf_text(prompt: str, max_tokens: int, model_id: str) -> str:
    """Text-only call via the HF backend (no image)."""
    from .hf import _load
    import torch
    model, processor = _load(model_id_override=model_id)
    inputs = processor(text=prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    generated = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0]


def _call_hf_vision(prompt: str, image: Image.Image, max_tokens: int, model_id: str) -> str:
    """Vision+text call via the HF backend.

    This path is the task-agnostic entrypoint used by
    :func:`frameselect.vlm.client.call_vision`. Unlike the MCQ path in
    :func:`frameselect.vlm.hf.call_vlm` (default ``parse_mcq=True``),
    we want the raw generated text here so callers can do captioning,
    free-form VQA, RAG rerank prompts, etc.
    """
    from .hf import call_vlm
    msgs = [{"role": "user", "content": [image], "_text": prompt}]
    return call_vlm(
        msgs, model_id=model_id,
        parse_mcq=False, max_new_tokens=max_tokens,
    )


# -----------------------------------------------------------------------------
# Anthropic API path (SDK)
# -----------------------------------------------------------------------------
_anthropic_client = None


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        import os

        import anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set.")
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


def _call_anthropic(messages: list[dict], max_tokens: int, model_id: str) -> str:
    client = _get_anthropic_client()
    resp = client.messages.create(
        model=model_id,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0,
    )
    return resp.content[0].text


# -----------------------------------------------------------------------------
# Bedrock path
# -----------------------------------------------------------------------------
def _call_bedrock(messages: list[dict], max_tokens: int, model_id: str) -> str:
    """Call Bedrock, retrying in the next configured region on error.

    Each call starts at the next region in the cycle and uses that
    region's inference profile. On error we try the next region (the same
    one again when only one is configured), with a pause after throttling,
    for at least 3 attempts before giving up.
    """
    global _ep_cycle
    import time as _time
    from .bedrock import (
        _bedrock_clients, _build_payload, _canonical_model_name,
        _get_all_endpoints, _get_client, _parse_model_response, get_profile_for,
    )
    endpoints = _get_all_endpoints()
    if not endpoints:
        raise RuntimeError("No Bedrock regions configured.")

    with _ep_lock:
        if _ep_cycle is None:
            _ep_cycle = itertools.cycle(endpoints)

    # Claude or Nova request body, by the model's catalog entry.
    vendor, payload = _build_payload(_canonical_model_name(model_id) or "", messages)
    if vendor == "amazon":
        payload["inferenceConfig"]["maxTokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens
        payload.pop("top_k", None)
    n_attempts = max(3, min(4, len(endpoints)))
    last_err = None
    for attempt in range(n_attempts):
        with _ep_lock:
            source, region = next(_ep_cycle)
        client = _get_client(source, region)
        if client is None:
            continue

        # This model's profile for the region, as the MCQ answering path
        # resolves it; a model outside the catalog is sent as given.
        effective_model_id = get_profile_for(model_id, region) or model_id
        try:
            resp = client.invoke_model(
                modelId=effective_model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(payload),
            )
            result = json.loads(resp["body"].read())
            text = _parse_model_response(result, vendor)
            if not text:
                raise RuntimeError("empty Bedrock response")
            return text
        except Exception as e:
            last_err = e
            if _error_code(e) == "ThrottlingException":
                _time.sleep(2)
            # Refresh the cached client on expired credentials. Match the
            # explicit Bedrock error code rather than substring-sniffing a
            # getattr chain: read the ``Error.Code`` off the response dict
            # (present on botocore ClientError) and compare to the known
            # expired-credential codes.
            if _error_code(e) in _EXPIRED_ERROR_CODES:
                key = (source, region)
                _bedrock_clients.pop(key, None)
            continue
    logger.warning("All Bedrock regions failed after %d attempts (%s)",
                   n_attempts, type(last_err).__name__ if last_err else "none")
    raise RuntimeError(f"All Bedrock regions failed; last error: {last_err}")


def _error_code(exc: Exception) -> str:
    """Extract a botocore-style error code from an exception, or ``""``.

    Reads ``exc.response["Error"]["Code"]`` defensively — works for real
    ``botocore.exceptions.ClientError`` instances and any duck-typed
    exception carrying a dict ``response``.
    """
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        err = response.get("Error")
        if isinstance(err, dict):
            return err.get("Code", "") or ""
    return ""


# -----------------------------------------------------------------------------
# Model-ID normalization (mirror of the evaluator's model-ID normalisation)
# -----------------------------------------------------------------------------
def _normalize_model_id(model_id: Optional[str], backend: Optional[str] = None) -> str:
    """Rewrite a model ID to match the active backend's naming convention.

    ``backend`` defaults to the module-level ``_BACKEND`` for back-compat
    with old callers that didn't pass one.
    """
    if backend is None:
        backend = _BACKEND
    if model_id is None:
        return _DEFAULT_MODEL_ID
    looks_bedrock = model_id.startswith("us.anthropic.") or model_id.endswith("-v1:0")
    if backend == "anthropic" and looks_bedrock:
        m = model_id
        if m.startswith("us.anthropic."):
            m = m[len("us.anthropic."):]
        if m.endswith("-v1:0"):
            m = m[: -len("-v1:0")]
        return m
    if backend == "bedrock" and not looks_bedrock:
        return f"us.anthropic.{model_id}-v1:0"
    return model_id


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def call_text(
    prompt: str,
    max_tokens: int = 256,
    model_id: Optional[str] = None,
    *,
    vlm_spec: Optional[str] = None,
) -> str:
    """Text-only VLM call. Returns raw string.

    Parameters
    ----------
    prompt : str
        The text prompt.
    max_tokens : int
        Maximum tokens to generate.
    model_id : str, optional
        Override the backend's default model. Takes precedence over
        the model component of ``vlm_spec``.
    vlm_spec : str, optional
        Per-call ``"backend:model"`` override. When passed, this fully
        determines the backend + default model without touching any
        global state. VLM-based selectors pass their ``self._vlm_spec``
        here so each selector can route to its own VLM. When omitted, falls back to env-var detection.
    """
    backend, default_model = _resolve_selection_backend(vlm_spec)
    model_id = _normalize_model_id(model_id or default_model, backend=backend)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    if backend == "openai":
        return _call_openai(messages, max_tokens, model_id)
    if backend == "hf":
        return _call_hf_text(prompt, max_tokens, model_id)
    if backend == "bedrock":
        return _call_bedrock(messages, max_tokens, model_id)
    return _call_anthropic(messages, max_tokens, model_id)


def call_vision(
    prompt: str,
    image: Image.Image,
    max_tokens: int = 256,
    model_id: Optional[str] = None,
    max_image_size: int = 1024,
    *,
    vlm_spec: Optional[str] = None,
) -> str:
    """Vision+text VLM call with one PIL image. Returns raw string.

    Parameters
    ----------
    prompt : str
        The text prompt to pair with the image.
    image : PIL.Image.Image
        A single image.
    max_tokens : int
        Maximum tokens to generate.
    model_id : str, optional
        Override the backend's default model.
    max_image_size : int
        Longest-edge resize budget (JPEG-encoded before sending).
    vlm_spec : str, optional
        Per-call ``"backend:model"`` override — see :func:`call_text`.
    """
    backend, default_model = _resolve_selection_backend(vlm_spec)
    model_id = _normalize_model_id(model_id or default_model, backend=backend)
    if backend == "openai":
        b64 = _pil_to_b64(image, max_size=max_image_size)
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}]
        return _call_openai(messages, max_tokens, model_id)
    if backend == "hf":
        return _call_hf_vision(prompt, image, max_tokens, model_id)
    b64 = _pil_to_b64(image, max_size=max_image_size)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if backend == "bedrock":
        return _call_bedrock(messages, max_tokens, model_id)
    return _call_anthropic(messages, max_tokens, model_id)


def get_backend() -> str:
    """Return the default backend chosen from the environment at import time."""
    return _BACKEND
