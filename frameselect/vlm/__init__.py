# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""VLM client + backend wrappers.

Submodules:

* :mod:`frameselect.vlm.client`    — shared thin client used by selectors
  that internally call a VLM. Accepts a per-call
  ``vlm_spec="backend:model"`` argument so each selector routes its own
  VLM calls; no global state.
* :mod:`frameselect.vlm.openai`    — OpenAI-compatible backend (vLLM, Ollama, OpenAI)
* :mod:`frameselect.vlm.hf`        — local HuggingFace backend (any ``AutoModelForImageTextToText``)
* :mod:`frameselect.vlm.bedrock`   — AWS Bedrock backend
* :mod:`frameselect.vlm.anthropic` — Anthropic API direct
"""

from .server import VLMServer

__all__ = ["VLMServer"]
