# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""HuggingFace Transformers VLM backend.

Loads any vision-language model from the HuggingFace Hub locally and
runs inference in-process. Supports any model that:

1. Has a ``processor`` with ``apply_chat_template`` (Qwen2.5-VL,
   InternVL3, LLaVA-NeXT, Ovis, MiniCPM-V, Idefics3, etc.)
2. Accepts multi-image inputs via the processor's image API.

Activation
----------
Set the environment variable:

    export HF_VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
    # optional:
    export HF_VLM_DEVICE=cuda          # default: cuda if available, else cpu
    export HF_VLM_DTYPE=bfloat16       # default: bfloat16 on CUDA, float16 on MPS, float32 on CPU
    export HF_VLM_MAX_NEW_TOKENS=1     # default: 1 (MCQ only needs one token)
    export HF_VLM_LOAD_IN_4BIT=1       # optional 4-bit quantization

The evaluator auto-detects this backend when ``HF_VLM_MODEL`` is set
and ``OPENAI_BASE_URL`` is not set.

Notes
-----
- The model is loaded lazily on the first call and cached for the
  process lifetime.
- For MCQ tasks ``max_new_tokens=1`` is sufficient and fast.
- 4-bit quantization (``HF_VLM_LOAD_IN_4BIT=1``) requires
  ``bitsandbytes``: ``pip install bitsandbytes``.
- Multi-GPU: set ``HF_VLM_DEVICE=auto`` to use ``device_map="auto"``,
  which spreads the model across all GPUs.
"""

from __future__ import annotations

import io
import logging
import os
import re
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _model_id() -> str:
    """Get the HF model ID from ``HF_VLM_MODEL``. Callers that want an
    explicit override pass ``model_id=...`` at the call site."""
    return os.environ.get("HF_VLM_MODEL", "")


def _device() -> str:
    """Pick the device for HF model loading.

    * ``HF_VLM_DEVICE`` env var wins if set.
    * Otherwise: CUDA when available, CPU everywhere else.

    We deliberately do **not** auto-select MPS: the Qwen2-VL family
    uses ``Conv3d`` in the vision encoder and PyTorch's MPS backend
    has no ``Conv3d`` kernel (tracked upstream since 2023). macOS
    users get CPU — slow but correct. Set ``HF_VLM_DEVICE=mps``
    explicitly if you want to experiment with MPS on architectures
    that don't need Conv3d, or pair it with
    ``PYTORCH_ENABLE_MPS_FALLBACK=1`` for a per-op CPU fallback.
    """
    override = os.environ.get("HF_VLM_DEVICE")
    if override:
        return override
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _dtype_str() -> str:
    """Default dtype depends on what the active device actually supports.

    * CUDA     → bfloat16
    * MPS      → float16  (Apple Metal still has no bf16 path)
    * CPU      → float32  (bf16 CPU ops are slow / partial)

    Override with the ``HF_VLM_DTYPE`` env var when the auto-pick
    isn't what you want.
    """
    override = os.environ.get("HF_VLM_DTYPE")
    if override:
        return override
    try:
        import torch
        if torch.cuda.is_available():
            return "bfloat16"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "float16"
    except Exception:
        pass
    return "float32"


def _max_new_tokens() -> int:
    return int(os.environ.get("HF_VLM_MAX_NEW_TOKENS", "1"))


def _load_in_4bit() -> bool:
    return os.environ.get("HF_VLM_LOAD_IN_4BIT", "0") == "1"


def _trust_remote_code() -> bool:
    """Whether to allow HuggingFace to execute model-repo Python code.

    ``trust_remote_code=True`` lets a model repo run arbitrary Python at
    load time — remote code execution if the model id is ever attacker
    controlled. It is therefore **opt-in**: set ``FRAMESELECT_TRUST_REMOTE_CODE``
    to ``1``/``true``/``yes`` to enable it, and only ever for model ids
    you trust. Defaults to ``False``.

    Some models (InternVL, MiniCPM-V, Ovis, …) ship custom modules and
    will not load without it; the error message names this flag.
    """
    val = os.environ.get("FRAMESELECT_TRUST_REMOTE_CODE", "").strip().lower()
    return val in ("1", "true", "yes", "on")


# Backwards-compat alias expected by base.py's import block.
HAIKU_MODEL_ID = _model_id()


# ---------------------------------------------------------------------------
# Lazy model + processor
# ---------------------------------------------------------------------------

_model = None
_processor = None
_loaded_model_id: Optional[str] = None


def _load(model_id_override: Optional[str] = None):
    global _model, _processor, _loaded_model_id
    mid = model_id_override or _model_id()
    if not mid:
        raise ValueError(
            "No HF model specified. Pass vlm='hf:model-name' to the "
            "selector or evaluator (or set HF_VLM_MODEL env var)."
        )
    if _model is not None and _loaded_model_id == mid:
        return _model, _processor

    try:
        import torch
        from transformers import AutoProcessor
        # transformers ≥5.x uses AutoModelForImageTextToText for VLMs;
        # fall back to AutoModelForCausalLM for older versions.
        try:
            from transformers import AutoModelForImageTextToText as _AutoVLM
        except ImportError:
            try:
                from transformers import AutoModelForVision2Seq as _AutoVLM
            except ImportError:
                from transformers import AutoModelForCausalLM as _AutoVLM
    except ImportError as e:
        raise ImportError(
            "transformers and torch are required for the HuggingFace VLM backend.\n"
            "Install with: pip install transformers torch"
        ) from e

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(_dtype_str(), torch.bfloat16)
    device = _device()

    # Transformers 4.x expects ``torch_dtype=``; 5.x renamed it to
    # ``dtype=`` and rejects the old name on some model classes. Dispatch
    # by transformers version so this works across both lines (Intel
    # macOS is pinned to 4.52.x; Linux / Apple Silicon get the latest).
    import transformers as _tx
    _tx_major = int(_tx.__version__.split(".")[0])
    _dtype_kw = "dtype" if _tx_major >= 5 else "torch_dtype"

    trust_remote_code = _trust_remote_code()
    load_kwargs: dict = {
        _dtype_kw: dtype,
        "trust_remote_code": trust_remote_code,
    }
    if device == "auto":
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = device

    if _load_in_4bit():
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
            )
        except ImportError:
            raise ImportError(
                "bitsandbytes is required for 4-bit quantization.\n"
                "Install with: pip install bitsandbytes"
            )

    print(f"[frameselect.vlm.hf] Loading {mid} (dtype={_dtype_str()}, device={device})...")
    # Force the slow (numpy-based) image processor. The fast (torch-based)
    # processor in transformers 4.53+ calls ``torch.compiler.is_compiling()``
    # during every resize — an API that only exists on torch>=2.3. Intel
    # macOS is stuck at torch 2.2.2 and hits AttributeError there; the slow
    # processor sidesteps the whole path and produces equivalent output.
    _processor = AutoProcessor.from_pretrained(
        mid, trust_remote_code=trust_remote_code, use_fast=False
    )
    _model = _AutoVLM.from_pretrained(mid, **load_kwargs)
    _model.eval()
    _loaded_model_id = mid
    print(f"[frameselect.vlm.hf] {mid} loaded.")
    return _model, _processor


# ---------------------------------------------------------------------------
# Message building (same interface as frameselect.vlm.openai)
# ---------------------------------------------------------------------------

def build_mcq_message(
    frames: np.ndarray,
    question: str,
    options: list[str],
    *,
    image_size: Optional[int] = None,
) -> list[dict]:
    """Build a chat message list with PIL images embedded.

    Returns the same ``list[dict]`` shape as the OpenAI backend so
    the evaluator can call either backend transparently. The images
    are stored as PIL.Image objects (not base64) because the HF
    processor handles encoding internally.

    ``image_size`` is the evaluator's longest-edge budget: frames whose
    longer side exceeds it are downscaled (aspect ratio preserved),
    smaller frames are left as is. ``None`` keeps native resolution and
    leaves resizing to the model's processor.
    """
    from ._mcq_parse import mcq_answer_instruction

    pil_images = [Image.fromarray(f) for f in frames]
    if image_size is not None:
        resized = []
        for img in pil_images:
            w, h = img.size
            scale = image_size / max(w, h)
            if scale < 1.0:
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            resized.append(img)
        pil_images = resized
    option_str = "\n".join(options)
    text = (
        "These are frames sampled from a video. "
        "Answer the following multiple choice question.\n\n"
        f"Question: {question}\n"
        f"Options:\n{option_str}\n\n"
        f"{mcq_answer_instruction(len(options))}"
    )
    # Store PIL images in the message dict; call_vlm will extract them.
    # ``_n_options`` lets the row wrapper widen the parser's letter range.
    return [{
        "role": "user",
        "content": pil_images,
        "_text": text,
        "_n_options": len(options),
    }]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def call_vlm(
    messages: list[dict],
    model_id: Optional[str] = None,
    max_retries: int = 3,
    *,
    parse_mcq: bool = True,
    max_new_tokens: Optional[int] = None,
    n_options: Optional[int] = None,
) -> str:
    """Run local HF inference.

    Parameters
    ----------
    messages : list[dict]
        Single-turn message in the library's internal shape
        ``[{"role": "user", "content": [PIL.Image, ...], "_text": str}]``.
    model_id : str, optional
        Override the module-level default model id.
    max_retries : int
        How many retries on transient CUDA / processor errors.
    parse_mcq : bool, default True
        When True (the historical path used by MCQ evaluators) the raw
        text output is reduced to a single option letter via
        :func:`_mcq_parse.parse_mcq_letter`. Set to ``False`` for
        open-ended tasks (captioning, free-form VQA) where you want the
        full generated string back.
    max_new_tokens : int, optional
        Per-call override of the default token budget (env var
        ``HF_VLM_MAX_NEW_TOKENS``). Useful for captioning / free-form
        tasks that want more than the MCQ-sized 1-token default.
    n_options : int, optional
        Number of MCQ options. Widens the parser's valid-letter range so
        5--7-way benchmarks can return ``E``/``F``/``G``. Falls back to
        the message's ``_n_options`` (set by :func:`build_mcq_message`),
        then to ``A..D``.
    """
    import torch

    from ._mcq_parse import FAILED_ANSWER, parse_mcq_letter

    model, processor = _load(model_id_override=model_id)

    # Extract PIL images and text from our message format.
    msg = messages[0]
    pil_images: list[Image.Image] = msg["content"]
    text: str = msg.get("_text", "")
    if n_options is None:
        n_options = msg.get("_n_options")

    # Build the chat template. We try the standard HF chat template
    # first; fall back to a plain text prompt if the model doesn't
    # support it.
    try:
        # Standard multi-image chat template (Qwen2.5-VL, InternVL3, etc.)
        chat = [
            {
                "role": "user",
                "content": (
                    [{"type": "image"} for _ in pil_images]
                    + [{"type": "text", "text": text}]
                ),
            }
        ]
        prompt = processor.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=prompt,
            images=pil_images,
            return_tensors="pt",
        )
    except Exception:
        # Fallback: plain text + images without chat template
        prompt = text
        inputs = processor(
            text=prompt,
            images=pil_images,
            return_tensors="pt",
        )

    # Move to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    tok_budget = max_new_tokens if max_new_tokens is not None else _max_new_tokens()

    for attempt in range(max_retries):
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=tok_budget,
                    do_sample=False,
                )
            # Decode only the newly generated tokens
            input_len = inputs["input_ids"].shape[1]
            generated = output_ids[:, input_len:]
            text_out = processor.batch_decode(
                generated, skip_special_tokens=True
            )[0]
            if parse_mcq:
                return parse_mcq_letter(text_out, n_options=n_options)
            return text_out
        except Exception as e:
            if attempt < max_retries - 1:
                import time
                time.sleep(2 ** attempt)
            else:
                logger.warning("HF VLM inference failed: %s",
                               type(e).__name__)
                return FAILED_ANSWER if parse_mcq else ""

    return FAILED_ANSWER if parse_mcq else ""


def call_vlm_row(row, model_id: Optional[str] = None) -> str:
    """Row-wise wrapper for pandas.

    Threads the per-question option count through so the parser uses the
    real letter range for 5--7-way benchmarks. Prefers the row's
    ``options`` (evaluator path); the message's ``_n_options`` is the
    fallback used when only ``messages`` is available.
    """
    n_options = None
    try:
        opts = row["options"]
        n_options = len(opts) if opts is not None else None
    except (KeyError, TypeError):
        n_options = None
    # Only thread n_options when known, so the historical call signature
    # (messages, model_id) is preserved when a row carries no options.
    if n_options is not None:
        return call_vlm(row["messages"], model_id=model_id, n_options=n_options)
    return call_vlm(row["messages"], model_id=model_id)


# Aliases for back-compat with base.py import block.
call_haiku = call_vlm
call_haiku_row = call_vlm_row
