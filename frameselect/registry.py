# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Global registry of frame selection methods."""

from __future__ import annotations

import logging
from typing import Dict, Type

from .base import BaseSelector

logger = logging.getLogger(__name__)

_REGISTRY: Dict[str, Type[BaseSelector]] = {}


def register(name: str):
    """Decorator to register a selector class under *name*.

    Registering a name that is already taken logs a warning and
    overwrites the previous binding. A silent overwrite would let a
    stray duplicate ``@register("uniform")`` shadow the real selector
    with no signal, so surface it.
    """

    def wrapper(cls: Type[BaseSelector]):
        existing = _REGISTRY.get(name)
        if existing is not None and existing is not cls:
            logger.warning(
                "register: name %r already bound to %s; overwriting with %s",
                name,
                existing.__name__,
                cls.__name__,
            )
        _REGISTRY[name] = cls
        return cls

    return wrapper


def list_methods() -> list[str]:
    """Return sorted list of registered method names."""
    return sorted(_REGISTRY.keys())


def get_method(name: str) -> Type[BaseSelector]:
    """Look up a registered method by name."""
    if name not in _REGISTRY:
        avail = ", ".join(list_methods())
        raise ValueError(f"Unknown method '{name}'. Available: {avail}")
    return _REGISTRY[name]
