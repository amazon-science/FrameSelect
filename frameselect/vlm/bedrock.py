# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Claude and Amazon Nova models through AWS Bedrock.

Two models are registered by default:

- ``claude-haiku-4-5`` (the default)
- ``nova-2-lite``

Either a short catalog key or a full Bedrock inference-profile ID can be
passed. For each region the call goes to that region's inference profile
(``us.*`` / ``eu.*`` / ``jp.*``, or ``global.*`` elsewhere).

Credentials and regions
-----------------------

Clients are built with boto3's default credential chain (an instance
role, ``~/.aws/credentials``, ``AWS_ACCESS_KEY_ID`` /
``AWS_SECRET_ACCESS_KEY``, or ``AWS_PROFILE``); no ``.env`` file is
required. Requests go to ``AWS_REGION`` / ``AWS_DEFAULT_REGION``
(``us-east-1`` when neither is set), or to the regions listed in
``BEDROCK_REGIONS``::

    BEDROCK_REGIONS=us-east-1,us-west-2    # explicit list
    # or
    AWS_REGION=us-east-1                   # single region

A failed request is retried in the next listed region, with a short
pause after throttling.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from time import sleep
from typing import Optional

import boto3
import numpy as np
from botocore.exceptions import ClientError
from PIL import Image

logger = logging.getLogger(__name__)


# Error codes that indicate expired/invalid credentials — the cached
# client for that region must be dropped so a fresh one is built.
_EXPIRED_ERROR_CODES = frozenset({
    "ExpiredToken",
    "ExpiredTokenException",
    "RequestExpired",
})


# ---------------------------------------------------------------------------
# Model catalog — regional inference profiles per supported model
# ---------------------------------------------------------------------------

# Regions with a known inference-profile prefix, grouped by geography so
# the profile lookup below can pick the right regional profile.
_US_REGIONS    = ("us-east-1", "us-east-2", "us-west-2")
_EU_REGIONS    = ("eu-west-1", "eu-west-3", "eu-central-1", "eu-north-1")
_JP_REGIONS    = ("ap-northeast-1",)
_GLOBAL_REGIONS = (
    "ap-southeast-1", "ap-southeast-2", "ap-south-1",
    "ca-central-1", "sa-east-1",
)

ALL_REGIONS = _US_REGIONS + _EU_REGIONS + _JP_REGIONS + _GLOBAL_REGIONS


def _profiles_for(vendor: str, model_suffix: str) -> dict[str, str]:
    """Build ``region → profile`` for a given ``<vendor>.<model_suffix>``.

    ``vendor`` is the Bedrock namespace (``"anthropic"``, ``"amazon"``,
    …); ``model_suffix`` is the part after it (e.g.
    ``claude-haiku-4-5-20251001-v1:0`` or ``nova-2-lite-v1:0``). We
    prepend the right geo prefix per region.
    """
    prof: dict[str, str] = {}
    for r in _US_REGIONS:
        prof[r] = f"us.{vendor}.{model_suffix}"
    for r in _EU_REGIONS:
        prof[r] = f"eu.{vendor}.{model_suffix}"
    for r in _JP_REGIONS:
        prof[r] = f"jp.{vendor}.{model_suffix}"
    for r in _GLOBAL_REGIONS:
        prof[r] = f"global.{vendor}.{model_suffix}"
    return prof


# Models registered by default. Add a model by appending one entry here;
# the profile lookup picks it up automatically.
MODEL_CATALOG: dict[str, dict[str, str]] = {
    "claude-haiku-4-5":  _profiles_for("anthropic", "claude-haiku-4-5-20251001-v1:0"),
    "nova-2-lite":       _profiles_for("amazon",    "nova-2-lite-v1:0"),
}

# Default model.
DEFAULT_MODEL = "claude-haiku-4-5"
HAIKU_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"  # legacy alias


def _canonical_model_name(model_id: str) -> Optional[str]:
    """Map a full profile ID (e.g. ``us.anthropic.claude-haiku-...-v1:0``
    or ``global.amazon.nova-2-lite-v1:0``) or a short name
    (``claude-haiku-4-5``, ``nova-2-lite``) back to the catalog key,
    or ``None`` if it isn't in the catalog.
    """
    if model_id in MODEL_CATALOG:
        return model_id
    # Strip geo + vendor prefix if present. We recognise both anthropic
    # (Claude) and amazon (Nova) namespaces.
    short = model_id
    for geo in ("us.", "eu.", "jp.", "global."):
        for vendor in ("anthropic.", "amazon."):
            prefix = geo + vendor
            if short.startswith(prefix):
                short = short[len(prefix):]
                break
        else:
            continue
        break
    # Strip trailing ``-v<N>:<M>`` suffix.
    short = re.sub(r"-v\d+:\d+$", "", short)
    # Strip trailing YYYYMMDD date if present.
    short = re.sub(r"-\d{8}$", "", short)
    return short if short in MODEL_CATALOG else None


def get_profile_for(model: str, region: str) -> Optional[str]:
    """Return the Bedrock profile ID to send for ``(model, region)``, or
    ``None`` if the model isn't in the catalog or the region isn't
    covered. The call moves on to the next region in that case.
    """
    canon = _canonical_model_name(model)
    if canon is None:
        return None
    return MODEL_CATALOG[canon].get(region)


# Region -> profile map for the default model (read by
# :mod:`frameselect.vlm.client`).
_REGION_TO_PROFILE: dict[str, str] = dict(MODEL_CATALOG[DEFAULT_MODEL])


# ---------------------------------------------------------------------------
# Regions and clients
# ---------------------------------------------------------------------------

def _resolve_regions() -> list[str]:
    """Regions to send requests to.

    Priority:
    1. ``BEDROCK_REGIONS`` (comma-separated) -- explicit list.
    2. ``AWS_REGION`` / ``AWS_DEFAULT_REGION`` -- a single region.
    3. ``us-east-1``.
    """
    explicit = [r.strip() for r in os.getenv("BEDROCK_REGIONS", "").split(",") if r.strip()]
    if explicit:
        return explicit
    single = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    return [single or "us-east-1"]


ACTIVE_REGIONS = _resolve_regions()

# One client per region, keyed ``("default", region)`` so callers can
# unpack ``(credential source, region)`` endpoints.
_bedrock_clients: dict = {}


def _new_client(region: str):
    """Create a Bedrock runtime client from boto3's default credentials."""
    return boto3.client("bedrock-runtime", region_name=region)


def _init_clients():
    """Create the per-region clients once."""
    if _bedrock_clients:
        return
    for region in ACTIVE_REGIONS:
        _bedrock_clients[("default", region)] = _new_client(region)
    if os.getenv("FRAMESELECT_VERBOSE"):
        print(
            f"[frameselect.vlm.bedrock] {len(ACTIVE_REGIONS)} region(s): "
            f"{', '.join(ACTIVE_REGIONS)}; boto3 default credential chain."
        )


def _get_all_endpoints() -> list[tuple[str, str]]:
    """Return the ``("default", region)`` endpoints, one per region."""
    _init_clients()
    return list(_bedrock_clients.keys())


def _get_client(source: str, region: str):
    """Return the client for ``(source, region)``, rebuilding it if dropped."""
    key = (source, region)
    if key not in _bedrock_clients:
        _bedrock_clients[key] = _new_client(region)
    return _bedrock_clients.get(key)


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def _encode_frame(frame: np.ndarray, max_size: Optional[int] = 512) -> str:
    """Encode a frame as base64 JPEG, resized to fit within ``max_size``.

    ``max_size=None`` means "no resize" — the full-resolution JPEG is
    returned. This is the passthrough behaviour used by openai/hf
    backends; Bedrock falls back to 512 px if no value is configured.
    """
    img = Image.fromarray(frame)
    if max_size is None:
        max_size = 512  # Bedrock cap — protects against None leaking in
    ratio = min(max_size / img.width, max_size / img.height)
    if ratio < 1:
        img = img.resize((int(img.width * ratio), int(img.height * ratio)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Message building
# ---------------------------------------------------------------------------

def build_mcq_message(
    frames: np.ndarray,
    question: str,
    options: list[str],
    image_size: int = 512,
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


# ---------------------------------------------------------------------------
# API call with region retry
# ---------------------------------------------------------------------------

def _claude_to_nova_messages(messages: list[dict]) -> list[dict]:
    """Translate Claude message content blocks to Nova's invoke-model schema.

    Claude uses ``{"type": "image", "source": {"type": "base64",
    "media_type": "image/jpeg", "data": b64}}`` and ``{"type": "text",
    "text": "..."}``. Nova uses ``{"image": {"format": "jpeg",
    "source": {"bytes": b64}}}`` and ``{"text": "..."}``.
    """
    out: list[dict] = []
    for m in messages:
        new_content: list[dict] = []
        for c in m.get("content", []):
            ctype = c.get("type")
            if ctype == "image":
                src = c.get("source", {})
                new_content.append({
                    "image": {
                        "format": (src.get("media_type", "image/jpeg")
                                   .split("/")[-1]),
                        "source": {"bytes": src.get("data", "")},
                    }
                })
            elif ctype == "text":
                new_content.append({"text": c.get("text", "")})
            else:
                # Unknown block — pass through untouched.
                new_content.append(c)
        out.append({"role": m.get("role", "user"), "content": new_content})
    return out


def _parse_model_response(result: dict, vendor: str) -> str:
    """Pull the text output from either a Claude or Nova invoke-model
    response body."""
    if vendor == "amazon":
        # Nova: {"output": {"message": {"content": [{"text": "..."}]}}}
        try:
            blocks = result["output"]["message"]["content"]
            for b in blocks:
                if "text" in b:
                    return b["text"]
        except (KeyError, IndexError, TypeError):
            pass
        return ""
    # Anthropic (default): {"content": [{"type": "text", "text": "..."}]}
    try:
        return result["content"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return ""


def _build_payload(canon: str, messages: list[dict]) -> tuple[str, dict]:
    """Return ``(vendor, payload_dict)`` for the given model short name."""
    if canon.startswith("nova"):
        return "amazon", {
            "messages": _claude_to_nova_messages(messages),
            "inferenceConfig": {"maxTokens": 32, "temperature": 0, "topP": 1.0},
        }
    # Anthropic Claude — existing native payload.
    return "anthropic", {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": messages,
        "max_tokens": 32,
        "temperature": 0,
        "top_k": 1,
    }


def call_haiku_parallel(row, model_id: str = HAIKU_MODEL_ID):
    """Answer one evaluator row; retries in the next region on failure.

    ``model_id`` picks which model to call. It can be either:

    - a short catalog key (``"claude-haiku-4-5"``, ``"nova-2-lite"``), or
    - a full Bedrock inference-profile ID (``"us.anthropic.claude-...-v1:0"``
      or ``"us.amazon.nova-2-lite-v1:0"``).

    The caller's region determines which regional profile (us./eu./jp./
    global.) gets sent for that model. Regions not covered by the chosen
    model's catalog are skipped and the call moves to the next
    region. The payload schema + response parser are dispatched by
    vendor (``anthropic`` for Claude, ``amazon`` for Nova).
    """
    from ._mcq_parse import FAILED_ANSWER, parse_mcq_letter

    endpoints = _get_all_endpoints()
    if not endpoints:
        return FAILED_ANSWER

    canon = _canonical_model_name(model_id) or DEFAULT_MODEL
    vendor, payload = _build_payload(canon, row["messages"])

    # Widen the parser's valid-letter range for 5--7-way benchmarks.
    n_options = _row_n_options(row)

    total = len(endpoints)
    current_idx = row["region_index"] % total

    for attempt in range(total + 2):
        source, region = endpoints[current_idx % total]
        client = _get_client(source, region)
        if client is None:
            current_idx += 1
            continue

        # Pick the right profile for this (model, region) pair.
        effective_model_id = get_profile_for(canon, region)
        if effective_model_id is None:
            # Model isn't covered in this region — skip forward.
            current_idx += 1
            continue

        try:
            response = client.invoke_model(
                modelId=effective_model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(payload),
            )
            result = json.loads(response["body"].read())
            text = _parse_model_response(result, vendor)
            return parse_mcq_letter(text, n_options=n_options)

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            # Drop the cached client on expired credentials so the next
            # attempt rebuilds it.
            if error_code in _EXPIRED_ERROR_CODES:
                _bedrock_clients.pop((source, region), None)
            # All ClientErrors advance to the next region: a code such as
            # AccessDenied in one region may still succeed in another, so
            # we try the rest before giving up rather than returning a
            # fake answer.
            current_idx += 1
            if error_code == "ThrottlingException":
                sleep(2)
        except Exception:
            current_idx += 1
            sleep(1)

    # Every region failed — return the sentinel, NOT a fabricated "A".
    logger.warning(
        "Bedrock call failed in all %d region(s) for model=%s", total, canon
    )
    return FAILED_ANSWER


def _row_n_options(row) -> Optional[int]:
    """Best-effort option count from an evaluator row (``None`` if absent)."""
    try:
        opts = row["options"]
    except (KeyError, TypeError, IndexError):
        return None
    try:
        return len(opts) if opts is not None else None
    except TypeError:
        return None
